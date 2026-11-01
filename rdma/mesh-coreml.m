/* Prepared Core ML bindings shared by the FX and native frontends.
   Compilation, model loading and named view binding are preparation operations.
   Run invokes the already-bound model; no copied-output substitute exists. */
#include "mesh-coreml.h"
#include "mesh-tensor-layout.h"
#include "../etc/bridge/layout.h"
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

static NSMutableArray *models_;
static NSMutableDictionary *loaded_;
static NSMutableArray *placements_, *placement_text_;
static NSMutableArray *contracts_, *contract_text_;
static NSMutableArray *references_, *keys_, *model_resources_;

const char *mesh_coreml_error(void) { return mesh_host_error_text; }

// Walk nested blocks as well as the selected function's outer operations.
// This is the provider's anticipated placement; no prediction is submitted.
static NSString *device_name(id<MLComputeDeviceProtocol> device) API_AVAILABLE(macos(14.0)) {
  return [device isKindOfClass:MLNeuralEngineComputeDevice.class]?@"ane":
    [device isKindOfClass:MLGPUComputeDevice.class]?@"gpu":[device isKindOfClass:MLCPUComputeDevice.class]?@"cpu":@"unknown";
}
static NSArray *placement_block(MLComputePlan *plan,MLModelStructureProgramBlock *block,
    NSMutableDictionary *preferred,NSMutableDictionary *supported,NSMutableDictionary *unresolved)
    API_AVAILABLE(macos(14.4)) {
  NSMutableArray *operations=[NSMutableArray array];
  for(MLModelStructureProgramOperation *op in block.operations) {
    MLComputePlanDeviceUsage *usage=[plan computeDeviceUsageForMLProgramOperation:op];
    NSMutableArray *eligible=[NSMutableArray array],*outputs=[NSMutableArray array],*blocks=[NSMutableArray array];
    NSString *name=nil;
    if(usage) {
      name=device_name(usage.preferredComputeDevice);
      preferred[name]=@([preferred[name] unsignedLongValue]+1);
      for(id<MLComputeDeviceProtocol> device in usage.supportedComputeDevices) {
        NSString *kind=device_name(device);
        if(![eligible containsObject:kind])[eligible addObject:kind];
      }
      for(NSString *kind in eligible)supported[kind]=@([supported[kind] unsignedLongValue]+1);
    } else unresolved[op.operatorName]=@([unresolved[op.operatorName] unsignedLongValue]+1);
    for(MLModelStructureProgramNamedValueType *output in op.outputs)[outputs addObject:output.name];
    for(MLModelStructureProgramBlock *child in op.blocks)
      [blocks addObject:placement_block(plan,child,preferred,supported,unresolved)];
    [operations addObject:@{@"operator":op.operatorName,@"outputs":outputs,@"preferred":name?:NSNull.null,
      @"supported":usage?eligible:NSNull.null,@"blocks":blocks}];
  }
  return operations;
}
static NSDictionary *model_placement(NSURL *compiled,MLModelConfiguration *configuration,NSArray *execution_devices) {
  if(@available(macOS 15.0,*)) {
    dispatch_semaphore_t done=dispatch_semaphore_create(0);
    __block MLComputePlan *plan=nil;__block NSError *failure=nil;
    [MLComputePlan loadContentsOfURL:compiled configuration:configuration completionHandler:^(MLComputePlan *loaded,NSError *error) {
      plan=loaded;failure=error;dispatch_semaphore_signal(done);
    }];
    if(dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,(int64_t)MESH_PREPARATION_NS))) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"compute plan: preparation deadline of %.0f s",(double)MESH_PREPARATION_NS/1e9);return nil;
    }
    if(!plan){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"compute plan: %s",failure.localizedDescription.UTF8String);return nil;}
    MLModelStructureProgramFunction *function=plan.modelStructure.program.functions[configuration.functionName];
    if(!function){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"compute plan has no function %s",configuration.functionName.UTF8String);return nil;}
    NSMutableDictionary *preferred=[NSMutableDictionary dictionary],*supported=[NSMutableDictionary dictionary],
      *unresolved=[NSMutableDictionary dictionary];
    NSArray *operations=placement_block(plan,function.block,preferred,supported,unresolved);
    return @{@"allowed_devices":execution_devices,@"preferred":preferred,@"supported":supported,
      @"unresolved":unresolved,@"operations":operations};
  }
  snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"named Core ML compute plans require macOS 15");return nil;
}
static BOOL placement_allowed(NSDictionary *placement,NSArray *devices) {
  NSDictionary *preferred=placement[@"preferred"];
  NSMutableSet *outside=[NSMutableSet setWithArray:preferred.allKeys];[outside minusSet:[NSSet setWithArray:devices]];
  if(!outside.count)return YES;
  snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"Core ML preferred devices %s exceed declared devices %s",preferred.description.UTF8String,devices.description.UTF8String);
  return NO;
}

/* Core ML emits the MIL artifact. An explicitly selected direct ANE model
   consumes that artifact and publishes its compiled storage requirements. */
typedef BOOL (*MeshANEAction)(id,SEL,id,id,unsigned,NSError **);
typedef BOOL (*MeshANEMap)(id,SEL,id,id,BOOL,NSError **);
typedef void (*MeshANEUnmap)(id,SEL,id,id);
typedef BOOL (*MeshANEEvaluate)(id,SEL,id,id,id,unsigned,NSError **);
@interface MeshANEModel : NSObject
@property id model, client, requestResource;
@property NSDictionary *options, *contract, *capabilities;
@property MeshResourceBinding *declaredResources;
@property NSData *resources;
@property unsigned qos;
@property size_t capacity;
@property BOOL loaded, cacheMapping;
@property MeshANEAction unload;
@property MeshANEMap map;
@property MeshANEUnmap unmap;
@property MeshANEEvaluate evaluate;
@end
@implementation MeshANEModel
- (void)dealloc {
  if(_loaded){NSError *error=nil;_unload(_client,@selector(unloadModel:options:qos:error:),_model,_options,_qos,&error);}
}
@end

static void *ane_framework_;
static id ane_symbol(NSString *name) {
  CFStringRef *symbol=dlsym(ane_framework_,name.UTF8String);
  if(!symbol){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE capability has no symbol %s",name.UTF8String);return nil;}
  return (__bridge id)*symbol;
}
static BOOL ane_methods(id object,NSArray<NSString *> *names,NSDictionary *abi) {
  Class kind=object_isClass(object)?object:[object class];NSDictionary *declared=abi[NSStringFromClass(kind)];
  for(NSString *name in names) {
    Method method=object_isClass(object)?class_getClassMethod(kind,NSSelectorFromString(name)):class_getInstanceMethod(kind,NSSelectorFromString(name));
    NSString *encoding=method?@(method_getTypeEncoding(method)):nil;
    if(!encoding||![encoding isEqual:declared[name]]) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE %s.%s ABI %s differs from its declared capability %s",NSStringFromClass(kind).UTF8String,name.UTF8String,
        encoding.UTF8String?:"absent",[declared[name] UTF8String]?:"absent");return NO;
    }
  }
  return YES;
}
static const size_t model_element_[]={2,4,4,8,1};
static NSDictionary *ane_features(NSArray *live,NSArray *symbols,NSIndexSet *used,NSArray *alignment,
                                  NSDictionary *names,NSDictionary *capabilities) {
  NSMutableDictionary *result=[NSMutableDictionary dictionary],*bySymbol=[NSMutableDictionary dictionary];
  for(NSDictionary *feature in live)bySymbol[feature[@"Symbol"]]=feature;
  if(names.count!=live.count){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE needs an explicit name for every compiled feature");return nil;}
  NSMutableSet *assigned=[NSMutableSet set];
  for(NSString *name in names) {
    NSString *symbol=names[name];NSDictionary *feature=bySymbol[symbol];NSUInteger index=[symbols indexOfObject:symbol];
    if(!feature||index==NSNotFound||![used containsIndex:index]||[assigned containsObject:symbol]) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE feature %s has no distinct compiled symbol %s",name.UTF8String,symbol.UTF8String);return nil;
    }
    [assigned addObject:symbol];
    NSNumber *dtype=capabilities[@"types"][feature[@"Type"]];
    if(!dtype||dtype.unsignedIntegerValue>=MeshModelTypeCount){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE scalar type %s has no declared type mapping",[feature[@"Type"] UTF8String]);return nil;}
    const size_t element=model_element_[dtype.unsignedIntegerValue];
    for(NSString *key in capabilities[@"layout"])if(![feature[key] isEqual:capabilities[@"layout"][key]]) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"compiled ANE %s=%s differs from the declared layout decoder %s",key.UTF8String,[feature[key] description].UTF8String,[capabilities[@"layout"][key] description].UTF8String);return nil;
    }
    NSArray *dimensions=@[@"Batches",@"Channels",@"Depth",@"Height",@"Width"],
      *steps=@[@"BatchStride",@"PlaneStride",@"DepthStride",@"RowStride"];
    NSMutableArray *shape=[NSMutableArray array],*strides=[NSMutableArray array];
    for(NSString *axis in dimensions) {
      NSNumber *extent=feature[axis];if(![extent isKindOfClass:NSNumber.class]||extent.longLongValue<=0){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"compiled ANE needs its %s extent",axis.UTF8String);return nil;}
      [shape addObject:extent];
    }
    for(NSString *axis in steps) {
      NSNumber *stride=feature[axis];if(![stride isKindOfClass:NSNumber.class]||stride.longLongValue<=0||stride.unsignedLongLongValue%element){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"compiled ANE needs a scalar-aligned %s",axis.UTF8String);return nil;}
      [strides addObject:@(stride.unsignedLongLongValue/element)];
    }
    [strides addObject:@1];
    if(![feature[@"PlaneCount"] isKindOfClass:NSNumber.class]){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"compiled ANE needs its stored plane count");return nil;}
    const uint64_t bytes=[feature[@"PlaneStride"] unsignedLongLongValue]*[feature[@"PlaneCount"] unsignedLongLongValue];
    MetalTensorLayout *layout=[MetalTensorLayout shape:shape strides:strides offset:0 maps:@{} indices:nil];
    struct MetalTensorSpan span=layout.storageSpan;
    if(span.offset<0||((uint64_t)span.offset+span.count)*element>bytes){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"compiled ANE layout exceeds its declared stored planes");return nil;}
    const uint64_t boundary=[alignment[index] boolValue]?[capabilities[@"alignment_bytes"] unsignedLongLongValue]:element;
    result[name]=@{@"symbol":symbol,@"index":@(index),@"dtype":dtype,@"element":@(element),@"shape":shape,
      @"strides":strides,@"bytes":@(bytes),@"alignment":@(boundary),@"provider":feature};
  }
  return result;
}
static MeshANEModel *ane_model(NSURL *compiled,NSDictionary *given,MeshResourceBinding *declaredResources) {
  NSDictionary *declared=given[@"ane"],*capabilities=declared[@"capabilities"];
  if(![declared[@"model_key"] isKindOfClass:NSString.class]||![declared[@"model_type"] isKindOfClass:NSString.class]||
     ![declared[@"qos"] isKindOfClass:NSNumber.class]||![declared[@"cache_mapping"] isKindOfClass:NSNumber.class]||![declared[@"options"] isKindOfClass:NSDictionary.class]||
     ![declared[@"symbols"] isKindOfClass:NSDictionary.class]||![capabilities[@"types"] isKindOfClass:NSDictionary.class]||
     ![capabilities[@"layout"] isKindOfClass:NSDictionary.class]||![capabilities[@"abi"] isKindOfClass:NSDictionary.class]||
     ![capabilities[@"alignment_bytes"] isKindOfClass:NSNumber.class]||![capabilities[@"alignment_bytes"] unsignedLongLongValue]||
     ![capabilities[@"completion_arguments"] isEqual:@[@"BOOL",@"NSError"]]) {
    snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE requires model key/type, qos, cache mapping, options, symbol names and storage capabilities");return nil;
  }
  if(![given[@"devices"] isEqual:@[@"ane"]]){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE requires explicit ANE-only placement");return nil;}
  if(!ane_framework_)ane_framework_=dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_LOCAL|RTLD_LAZY);
  Class modelClass=NSClassFromString(@"_ANEModel"),clientClass=NSClassFromString(@"_ANEClient");
  if(!ane_framework_||!modelClass||!clientClass){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE framework is unavailable");return nil;}
  if(!ane_methods(modelClass,@[@"modelAtURL:key:"],capabilities[@"abi"])||!ane_methods(clientClass,@[@"sharedConnection"],capabilities[@"abi"]))return nil;
  id model=((id(*)(id,SEL,id,id))objc_msgSend)(modelClass,@selector(modelAtURL:key:),compiled,declared[@"model_key"]);
  id client=((id(*)(id,SEL))objc_msgSend)(clientClass,@selector(sharedConnection));
  if(!model||!client){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE model/client preparation failed");return nil;}
  if(!ane_methods(client,@[@"compileModel:options:qos:error:",@"loadModel:options:qos:error:",@"unloadModel:options:qos:error:",
      @"mapIOSurfacesWithModel:request:cacheInference:error:",@"unmapIOSurfacesWithModel:request:",@"evaluateWithModel:options:request:qos:error:"],capabilities[@"abi"])||
     !ane_methods(model,@[@"modelAttributes",@"inputSymbolIndicesForProcedureIndex:",@"outputSymbolIndicesForProcedureIndex:",@"queueDepth",@"program"],capabilities[@"abi"]))return nil;
  NSMutableDictionary *options=[NSMutableDictionary dictionary];
  for(NSString *key in declared[@"options"]){id symbol=ane_symbol(key);if(!symbol)return nil;options[symbol]=declared[@"options"][key];}
  id type=ane_symbol(declared[@"model_type"]),typeKey=ane_symbol(@"kANEFModelTypeKey");if(!type||!typeKey)return nil;options[typeKey]=type;
  MeshANEModel *prepared=[MeshANEModel new];prepared.client=client;prepared.model=model;prepared.options=options;prepared.capabilities=capabilities;prepared.qos=[declared[@"qos"] unsignedIntValue];prepared.cacheMapping=[declared[@"cache_mapping"] boolValue];
  prepared.unload=(MeshANEAction)[client methodForSelector:@selector(unloadModel:options:qos:error:)];
  prepared.map=(MeshANEMap)[client methodForSelector:@selector(mapIOSurfacesWithModel:request:cacheInference:error:)];
  prepared.unmap=(MeshANEUnmap)[client methodForSelector:@selector(unmapIOSurfacesWithModel:request:)];
  prepared.evaluate=(MeshANEEvaluate)[client methodForSelector:@selector(evaluateWithModel:options:request:qos:error:)];
  NSError *error=nil;
  MeshANEAction compile=(MeshANEAction)[client methodForSelector:@selector(compileModel:options:qos:error:)];
  if(!compile(client,@selector(compileModel:options:qos:error:),model,options,prepared.qos,&error)){
    snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE compile: %s",error.localizedDescription.UTF8String);return nil;
  }
  MeshANEAction load=(MeshANEAction)[client methodForSelector:@selector(loadModel:options:qos:error:)];
  if(!load(client,@selector(loadModel:options:qos:error:),model,options,prepared.qos,&error)){
    snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE load: %s",error.localizedDescription.UTF8String);return nil;
  }
  prepared.loaded=YES;
  prepared.requestResource=((id(*)(id,SEL))objc_msgSend)(model,@selector(program));
  const signed char capacity=((signed char(*)(id,SEL))objc_msgSend)(model,@selector(queueDepth));
  if(!prepared.requestResource||capacity<=0){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"compiled ANE model has no request resource/capacity");return nil;}
  prepared.capacity=(size_t)capacity;
  prepared.declaredResources=declaredResources;
  NSMutableData *resources=[NSMutableData dataWithBytes:prepared.declaredResources->entries
      length:prepared.declaredResources->count*sizeof(MeshHostResource)];
  const MeshHostResource request={(__bridge const void *)prepared.requestResource,prepared.capacity};
  [resources appendBytes:&request length:sizeof request];prepared.resources=resources;
  NSDictionary *attributes=((id(*)(id,SEL))objc_msgSend)(model,@selector(modelAttributes));
  id descriptionKey=ane_symbol(@"kANEFModelDescriptionKey"),procedureKey=ane_symbol(@"kANEFModelProcedureNameToIDMapKey");
  if(!descriptionKey||!procedureKey)return nil;
  NSDictionary *description=attributes[descriptionKey];NSNumber *procedure=description[procedureKey][given[@"function"]];
  if(!procedure){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"compiled ANE has no declared function %s",[given[@"function"] UTF8String]);return nil;}
  NSDictionary *network=nil;
  for(NSDictionary *candidate in attributes[@"NetworkStatusList"])if([candidate[@"Name"] isEqual:given[@"function"]]){network=candidate;break;}
  if(!network){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"compiled ANE has no storage schema for function %s",[given[@"function"] UTF8String]);return nil;}
  if([network[@"LiveStateList"] count]){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"ANE state must be functionalized into declared input/output pages before compilation");return nil;}
  NSMutableDictionary *contract=[@{@"backend":@"ane",@"output_storage":@"caller-pages",@"function":given[@"function"],@"procedure":procedure,@"request_capacity":@(prepared.capacity),
    @"resources":given[@"implementation"][@"resources"]} mutableCopy];
  NSArray *sides=@[@[@"inputs",@"LiveInputList",@"kANEFModelInputSymbolsArrayKey",@"kANEFModelInput16KAlignmentArrayKey",@"inputSymbolIndicesForProcedureIndex:"],
                   @[@"outputs",@"LiveOutputList",@"kANEFModelOutputSymbolsArrayKey",@"kANEFModelOutput16KAlignmentArrayKey",@"outputSymbolIndicesForProcedureIndex:"]];
  for(NSArray *side in sides) {
    id namesKey=ane_symbol(side[2]),alignmentKey=ane_symbol(side[3]);if(!namesKey||!alignmentKey)return nil;
    NSIndexSet *indices=((id(*)(id,SEL,unsigned))objc_msgSend)(model,NSSelectorFromString(side[4]),procedure.unsignedIntValue);
    NSDictionary *features=ane_features(network[side[1]],description[namesKey],indices,description[alignmentKey],declared[@"symbols"][side[0]],capabilities);
    if(!features)return nil;contract[side[0]]=features;
  }
  prepared.contract=contract;return prepared;
}
static NSDictionary *coreml_contract(MLModel *model,NSString *function) {
  if(@available(macOS 15.0,*))if(model.modelDescription.stateDescriptionsByName.count){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"Core ML state must be functionalized into declared input/output pages before compilation");return nil;}
  NSMutableDictionary *contract=[@{@"backend":@"coreml",@"output_storage":@"advisory",@"function":function} mutableCopy];
  for(NSArray *side in @[@[@"inputs",model.modelDescription.inputDescriptionsByName],@[@"outputs",model.modelDescription.outputDescriptionsByName]]) {
    NSMutableDictionary *features=[NSMutableDictionary dictionary];
    for(NSString *name in side[1]) {
      MLMultiArrayConstraint *type=[side[1][name] multiArrayConstraint];
      id dtype=NSNull.null,element=NSNull.null;
      const MLMultiArrayDataType types[]={MLMultiArrayDataTypeFloat16,MLMultiArrayDataTypeFloat32,MLMultiArrayDataTypeInt32,MLMultiArrayDataTypeDouble};
      for(NSUInteger i=0;i<sizeof types/sizeof *types;i++)if(type&&type.dataType==types[i]){dtype=@(i);element=@(model_element_[i]);}
      if(@available(macOS 26.0,*))if(type&&type.dataType==MLMultiArrayDataTypeInt8){dtype=@(MeshModelInt8);element=@(model_element_[MeshModelInt8]);}
      features[name]=@{@"shape":type.shape?:@[],@"strides":NSNull.null,@"bytes":NSNull.null,@"alignment":@1,@"dtype":dtype,@"element":element};
    }
    contract[side[0]]=features;
  }
  return contract;
}
long mesh_coreml_load(const char *package,const char *configuration_json) {
  @autoreleasepool { @try {
   NSDictionary *given=configuration_json?[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:configuration_json length:strlen(configuration_json)] options:0 error:nil]:nil;
   NSString *backend=given[@"backend"];
   if(![given isKindOfClass:NSDictionary.class]||![@[@"coreml",@"ane"] containsObject:backend]||
      ![given[@"units"] isKindOfClass:NSNumber.class]||![given[@"function"] isKindOfClass:NSString.class]||
      ![given[@"devices"] isKindOfClass:NSArray.class]||![given[@"devices"] count]||![given[@"state_pairs"] isKindOfClass:NSDictionary.class]) {
     snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"model requires explicit backend, units, function, devices and state_pairs");return -1;
   }
   NSArray *devices=given[@"devices"];NSString *function=given[@"function"];int units=[given[@"units"] intValue];
   const MLComputeUnits choices[]={MLComputeUnitsCPUAndNeuralEngine,MLComputeUnitsAll,MLComputeUnitsCPUOnly,MLComputeUnitsCPUAndGPU};
   NSArray *execution_devices=@[@[@"cpu",@"ane"],@[@"cpu",@"gpu",@"ane"],@[@"cpu"],@[@"cpu",@"gpu"]];
   if((unsigned)units>=sizeof choices/sizeof *choices){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"invalid compute units");return -1;}
   @synchronized([MLModel class]) {
    if(!models_){models_=[NSMutableArray new];loaded_=[NSMutableDictionary new];placements_=[NSMutableArray new];placement_text_=[NSMutableArray new];
      contracts_=[NSMutableArray new];contract_text_=[NSMutableArray new];references_=[NSMutableArray new];keys_=[NSMutableArray new];model_resources_=[NSMutableArray new];}
    NSArray *key=@[@(package),given];NSNumber *held=loaded_[key];
    if(held){long h=held.longValue;references_[h]=@([references_[h] unsignedLongValue]+1);return h;}
    MeshResourceBinding *resources=mesh_host_resources(given[@"implementation"][@"resources"]);if(!resources)return -1;
    NSError *error=nil;NSURL *compiled=[NSURL fileURLWithPath:@(package)];id model;NSDictionary *placement,*contract;
    if([backend isEqual:@"ane"]) {
      MeshANEModel *prepared=ane_model(compiled,given,resources);if(!prepared)return -1;model=prepared;contract=prepared.contract;
      placement=@{@"backend":@"ane",@"allowed_devices":@[@"ane"],@"preferred":@{},@"supported":@{},@"unresolved":@{},@"operations":@[]};
    } else {
      for(id device in devices)if(![@[@"cpu",@"gpu",@"ane"] containsObject:device]) {
        snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"unknown declared Core ML device %s",[device description].UTF8String);return -1;
      }
      MLModelConfiguration *configuration=[MLModelConfiguration new];configuration.computeUnits=choices[units];
      if(@available(macOS 15.0,*))configuration.functionName=function;
      else{snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"named Core ML functions require macOS 15");return -1;}
      placement=model_placement(compiled,configuration,execution_devices[units]);
      if(!placement||!placement_allowed(placement,devices))return -1;
      model=[MLModel modelWithContentsOfURL:compiled configuration:configuration error:&error];
      if(!model){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"load: %s",error.localizedDescription.UTF8String);return -1;}
      contract=coreml_contract(model,function);if(!contract)return -1;
    }
    for(NSString *name in given[@"state_pairs"]) {
      NSDictionary *pair=given[@"state_pairs"][name];
      if(![pair isKindOfClass:NSDictionary.class]||!contract[@"inputs"][pair[@"input"]]||!contract[@"outputs"][pair[@"output"]]) {
        snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"functional state %s requires declared input/output feature names",name.UTF8String);return -1;
      }
    }
    NSMutableDictionary *stateful=[contract mutableCopy];stateful[@"state_pairs"]=given[@"state_pairs"];
    stateful[@"resources"]=given[@"implementation"][@"resources"];contract=stateful;
    if([model isKindOfClass:MeshANEModel.class])[(MeshANEModel *)model setContract:contract];
    NSString *text=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:placement options:NSJSONWritingSortedKeys error:nil] encoding:NSUTF8StringEncoding];
    NSString *schema=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:contract options:NSJSONWritingSortedKeys error:nil] encoding:NSUTF8StringEncoding];
    NSUInteger handle=[models_ indexOfObject:NSNull.null];
    if(handle==NSNotFound){handle=models_.count;[models_ addObject:model];[placements_ addObject:placement];[placement_text_ addObject:text];
      [contracts_ addObject:contract];[contract_text_ addObject:schema];[references_ addObject:@1];[keys_ addObject:key];[model_resources_ addObject:resources];}
    else{models_[handle]=model;placements_[handle]=placement;placement_text_[handle]=text;contracts_[handle]=contract;contract_text_[handle]=schema;references_[handle]=@1;keys_[handle]=key;model_resources_[handle]=resources;}
    loaded_[key]=@(handle);return (long)handle;
   }
  } @catch(NSException *failure){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"model preparation: %s",failure.reason.UTF8String);return -1;} }
}

void mesh_coreml_release(long handle) {
  @synchronized([MLModel class]) {
    NSUInteger left=[references_[handle] unsignedLongValue]-1;references_[handle]=@(left);
    if(!left){[loaded_ removeObjectForKey:keys_[handle]];models_[handle]=[NSNull null];placements_[handle]=[NSNull null];placement_text_[handle]=[NSNull null];contracts_[handle]=[NSNull null];contract_text_[handle]=[NSNull null];keys_[handle]=[NSNull null];model_resources_[handle]=[NSNull null];}
  }
}

@interface MeshModelInvocation : NSObject { @public const char *(*prepare)(id,MeshHostCompletion,void *,MeshHostTask *); }
@end
@implementation MeshModelInvocation
@end

#include "mesh-ane-entry.inc"
static void model_task_release(void *context) { CFRelease(((MeshANEEntry *)context)->owner); }

@interface MeshANEInvocation : MeshModelInvocation { @public
  void *entry_library;
  void (*submit)(void *);
  MeshANEConfigure configure;
}
@property NSData *arguments;
@property long handle;
@property MeshANEModel *prepared;
@property NSArray *inputs,*outputs,*inputIndices,*outputIndices;
@property NSDictionary *surfaces;
@end
@implementation MeshANEInvocation
- (void)dealloc { if(entry_library)dlclose(entry_library);mesh_coreml_release(_handle); }
@end
@interface MeshANETask : NSObject { @public MeshANEEntry entry; }
@property MeshANEInvocation *source;
@property id request;
@property BOOL mapped;
@end
@implementation MeshANETask
- (void)dealloc {
  MeshANEModel *prepared=_source.prepared;
  if(_mapped)prepared.unmap(prepared.client,@selector(unmapIOSurfacesWithModel:request:),prepared.model,_request);
}
@end
static const char *ane_prepare(id object,MeshHostCompletion complete,void *context,MeshHostTask *result) {
  MeshANEInvocation *source=object;MeshANEModel *model=source.prepared;
  Class requestClass=NSClassFromString(@"_ANERequest");
  MeshANETask *task=[MeshANETask new];task.source=source;
  task.request=((id(*)(id,SEL,id,id,id,id,id))objc_msgSend)(requestClass,@selector(requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:),
    source.inputs,source.inputIndices,source.outputs,source.outputIndices,model.contract[@"procedure"]);
  if(!task.request)return "direct ANE request construction failed";
  if(!ane_methods(task.request,@[@"setCompletionHandler:"],model.capabilities[@"abi"]))return mesh_host_error_text;
  task->entry=(MeshANEEntry){.client=(__bridge void *)model.client,.model=(__bridge void *)model.model,
    .options=(__bridge void *)model.options,.request=(__bridge void *)task.request,.evaluate=model.evaluate,.qos=model.qos,
    .complete=complete,.completion_context=context,.arguments=source.arguments.bytes};
  source->configure(&task->entry);
  NSError *error=nil;task.mapped=model.map(model.client,@selector(mapIOSurfacesWithModel:request:cacheInference:error:),model.model,task.request,model.cacheMapping,&error);
  if(!task.mapped){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE page mapping: %s",error.localizedDescription.UTF8String);return mesh_host_error_text;}
  task->entry.owner=(__bridge_retained void *)task;
  *result=(MeshHostTask){.context=&task->entry,.submit=source->submit,.release=model_task_release,
    .resources=model.resources.bytes,.resource_count=model.resources.length/sizeof(MeshHostResource)};return NULL;
}
int mesh_coreml_entry(void *binding,const char *source,const char *compiler,const void *arguments,size_t bytes) {
  const char *path=mesh_cpu_compile(source,compiler);if(!path)return -1;
  void *library=dlopen(path,RTLD_NOW|RTLD_LOCAL);
  if(!library){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"ANE source library: %s",dlerror());return -1;}
  void (*submit)(void *)=dlsym(library,"mesh_ane_submit");
  MeshANEConfigure configure=dlsym(library,"mesh_ane_configure");
  if(!submit||!configure){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"ANE source entry: %s",dlerror());dlclose(library);return -1;}
  MeshANEInvocation *p=(__bridge MeshANEInvocation *)binding;
  if(p->entry_library)dlclose(p->entry_library);
  p->entry_library=library;p->submit=submit;p->configure=configure;
  p.arguments=[NSData dataWithBytes:arguments length:bytes];return 0;
}
static void *ane_bind(long handle,MeshANEModel *model,const MeshModelView *inputs,size_t ni,const MeshModelView *outputs,size_t no) {
  Class surfaceClass=NSClassFromString(@"_ANEIOSurfaceObject"),requestClass=NSClassFromString(@"_ANERequest");
  if(!ane_methods(surfaceClass,@[@"objectWithIOSurface:startOffset:"],model.capabilities[@"abi"])||
     !ane_methods(requestClass,@[@"requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:"],model.capabilities[@"abi"]))return NULL;
  CFStringRef *clientAddress=dlsym(RTLD_DEFAULT,"kIOSurfaceClientAddress");
  if(!clientAddress){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"IOSurface client-page wrapping is unavailable");return NULL;}
  NSMutableDictionary *surfaces=[NSMutableDictionary dictionary];
  NSMutableArray *in=[NSMutableArray array],*out=[NSMutableArray array],*inIndices=[NSMutableArray array],*outIndices=[NSMutableArray array];
  for(NSUInteger side=0;side<2;side++) {
    const MeshModelView *views=side?outputs:inputs;size_t count=side?no:ni;
    NSDictionary *features=model.contract[side?@"outputs":@"inputs"];NSMutableSet *names=[NSMutableSet set];
    if(count!=features.count){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE requires every declared %s feature",side?"output":"input");return NULL;}
    for(size_t i=0;i<count;i++) {
      const MeshModelView *view=views+i;NSString *name=view->name?@(view->name):nil;NSDictionary *feature=name?features[name]:nil;
      if(!feature||[names containsObject:name]){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE binding has no distinct declared feature");return NULL;}[names addObject:name];
      if(view->dtype!=[feature[@"dtype"] intValue]){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE %s has a different scalar type",view->name);return NULL;}
      NSMutableArray *shape=[NSMutableArray array],*steps=[NSMutableArray array];
      for(int j=0;j<view->rank;j++){[shape addObject:@(view->shape[j])];[steps addObject:@(view->strides[j])];}
      MetalTensorLayout *expected=[[MetalTensorLayout shape:feature[@"shape"] strides:feature[@"strides"] offset:0 maps:@{} indices:nil] reshape:shape];
      MetalTensorLayout *actual=[MetalTensorLayout shape:shape strides:steps offset:0 maps:@{} indices:nil];
      if(![actual equivalent:expected]){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE %s needs its prepared provider layout",view->name);return NULL;}
      const MeshModelPages pages=view->pages;const size_t bytes=[feature[@"bytes"] unsignedLongLongValue],alignment=[feature[@"alignment"] unsignedLongLongValue];
      if(!pages.base||pages.offset>pages.bytes||bytes>pages.bytes-pages.offset||((uintptr_t)pages.base+pages.offset)%alignment){
        snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"direct ANE %s needs its declared page footprint/alignment",view->name);return NULL;
      }
      NSArray *key=@[@((uintptr_t)pages.base),@(pages.bytes)];id surface=surfaces[key];
      if(!surface) {
        surface=CFBridgingRelease(IOSurfaceCreate((__bridge CFDictionaryRef)@{(__bridge id)*clientAddress:@((uintptr_t)pages.base),(__bridge id)kIOSurfaceAllocSize:@(pages.bytes)}));
        if(!surface||IOSurfaceGetBaseAddress((__bridge IOSurfaceRef)surface)!=pages.base||IOSurfaceGetAllocSize((__bridge IOSurfaceRef)surface)!=pages.bytes){
          snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"IOSurface did not bind the declared bridge pages");return NULL;
        }
        surfaces[key]=surface;
      }
      id operand=((id(*)(id,SEL,IOSurfaceRef,id))objc_msgSend)(surfaceClass,@selector(objectWithIOSurface:startOffset:),(__bridge IOSurfaceRef)surface,@(pages.offset));
      [(side?out:in) addObject:operand];[(side?outIndices:inIndices) addObject:feature[@"index"]];
    }
  }
  MeshANEInvocation *p=[MeshANEInvocation new];p->prepare=ane_prepare;p.handle=handle;p.prepared=model;
  p->submit=mesh_ane_submit;p->configure=mesh_ane_configure;
  p.inputs=in;p.outputs=out;p.inputIndices=inIndices;p.outputIndices=outIndices;p.surfaces=surfaces;
  @synchronized([MLModel class]){references_[handle]=@([references_[handle] unsignedLongValue]+1);}
  return (__bridge_retained void *)p;
}
void *mesh_coreml_bind(long handle,const MeshModelView *inputs,size_t ni,const MeshModelView *outputs,size_t no) {
  @autoreleasepool { @try {
    id model;
    @synchronized([MLModel class]) {
      if(handle<0||handle>=(long)models_.count||models_[handle]==NSNull.null){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"no model %ld",handle);return NULL;}
      model=models_[handle];
    }
    if(![contracts_[handle][@"output_storage"] isEqual:@"caller-pages"]) {
      snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"prepared page execution requires guaranteed caller-owned output pages; public Core ML outputBackings are advisory");return NULL;
    }
    return ane_bind(handle,model,inputs,ni,outputs,no);
  } @catch(NSException *failure){snprintf(mesh_host_error_text,sizeof mesh_host_error_text,"model binding: %s",failure.reason.UTF8String);return NULL;} }
}
const char *mesh_coreml_prepare(void *binding,MeshHostCompletion complete,void *context,MeshHostTask *task) {
  MeshModelInvocation *p=(__bridge MeshModelInvocation *)binding;return p->prepare(p,complete,context,task);
}
const char *mesh_coreml_contract(long handle) {
  @synchronized([MLModel class]){return [contract_text_[handle] UTF8String];}
}

void mesh_coreml_retain(void *binding) { CFRetain(binding); }
void mesh_coreml_unbind(void *binding) { CFRelease(binding); }

/* Placement is a property of the prepared model. Reading it never reloads a
   compute plan or waits for a provider callback. The handle owns the string. */
const char *mesh_coreml_placement(long handle) {
  @synchronized([MLModel class]) { return [placement_text_[handle] UTF8String]; }
}

/* A model package compiled (MLModel compileModelAtURL) into `destination` (a .mlmodelc, replaced where present), so a
   process loads it without compiling: 0, or -1. */
int mesh_coreml_compile(const char *package, const char *destination) {
  @autoreleasepool {
    NSError *error = nil;
    NSURL *compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:@(package)] error:&error];
    if (!compiled) { snprintf(mesh_host_error_text, sizeof mesh_host_error_text, "compile: %s", error.localizedDescription.UTF8String); return -1; }
    NSURL *target = [NSURL fileURLWithPath:@(destination)];
    [[NSFileManager defaultManager] removeItemAtURL:target error:nil];
    if (![[NSFileManager defaultManager] moveItemAtURL:compiled toURL:target error:&error]) {
      snprintf(mesh_host_error_text, sizeof mesh_host_error_text, "move: %s", error.localizedDescription.UTF8String); return -1;
    }
    return 0;
  }
}
