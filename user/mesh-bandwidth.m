#import <Foundation/Foundation.h>
#include <math.h>
#include <unistd.h>
extern CFDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
extern CFDictionaryRef IOReportCopyAllChannels(uint64_t, uint64_t);
extern CFStringRef IOReportChannelGetGroup(CFDictionaryRef);
extern CFStringRef IOReportChannelGetSubGroup(CFDictionaryRef);
extern CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
extern CFTypeRef IOReportCreateSubscription(CFTypeRef, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamples(CFTypeRef, CFMutableDictionaryRef, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern int IOReportStateGetCount(CFDictionaryRef);
extern CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef, int);
extern uint64_t IOReportStateGetResidency(CFDictionaryRef, int);

static void emit(NSDictionary *channel){
  if(!channel){ printf("null"); return; }
  CFDictionaryRef c=(__bridge CFDictionaryRef)channel;
  int n=IOReportStateGetCount(c);
  double total=0,lo=0,hi=0,prev=0;
  for(int i=0;i<n;i++){
    NSString *name=(__bridge NSString *)IOReportStateGetNameForIndex(c,i);
    double upper=name.doubleValue,weight=(double)IOReportStateGetResidency(c,i);
    total+=weight; lo+=weight*prev; hi+=weight*upper; prev=upper;
  }
  if(total<=0){ printf("null"); return; }
  lo/=total; hi/=total; prev=0;
  double vlo=0,vhi=0;
  for(int i=0;i<n;i++){
    NSString *name=(__bridge NSString *)IOReportStateGetNameForIndex(c,i);
    double upper=name.doubleValue,weight=(double)IOReportStateGetResidency(c,i);
    vlo+=weight*(prev-lo)*(prev-lo); vhi+=weight*(upper-hi)*(upper-hi); prev=upper;
  }
  printf("{\"lower_gbs\":%.6f,\"upper_gbs\":%.6f,\"lower_variance\":%.6f,\"upper_variance\":%.6f,\"lower_sd\":%.6f,\"upper_sd\":%.6f,\"residency\":%.0f}",lo,hi,vlo/total,vhi/total,sqrt(vlo/total),sqrt(vhi/total),total);
}

static void emit_delta(CFDictionaryRef deltaRef,int ms,NSString *group){
  @autoreleasepool {
    NSDictionary *delta=(__bridge NSDictionary *)deltaRef;
    NSArray *wanted=@[@"AMCC RD",@"AMCC WR",@"AMCC RD+WR",@"AGX RD",@"AGX WR",@"AGX RD+WR"];
    NSMutableDictionary *found=[NSMutableDictionary dictionary];
    for(NSDictionary *channel in delta[@"IOReportChannels"]){
      NSString *name=channel[@"LegendChannel"][2];
      if([wanted containsObject:name]) found[name]=channel;
    }
    printf("{\"up\":true,\"sample_ms\":%d,\"source\":\"IOReport/%s/DCS BW\"",ms,group.UTF8String);
    for(NSString *name in wanted){ printf(",\"%s\":",name.UTF8String); emit(found[name]); }
    printf("}\n"); fflush(stdout);
  }
}

int main(int argc,char **argv){
  @autoreleasepool {
    int streaming=argc>1&&strcmp(argv[1],"-s")==0;
    int ms=argc>2&&(strcmp(argv[1],"-i")==0||streaming)?atoi(argv[2]):250;
    if(argc>1&&strcmp(argv[1],"-l")==0){
      NSDictionary *all=CFBridgingRelease(IOReportCopyAllChannels(0,0));
      NSMutableArray *names=[NSMutableArray array];
      for(NSDictionary *channel in all[@"IOReportChannels"]){
        CFDictionaryRef c=(__bridge CFDictionaryRef)channel;
        NSString *group=(__bridge NSString *)IOReportChannelGetGroup(c);
        NSString *subgroup=(__bridge NSString *)IOReportChannelGetSubGroup(c);
        NSString *name=(__bridge NSString *)IOReportChannelGetChannelName(c);
        [names addObject:@[group?:@"",subgroup?:@"",name?:@""]];
      }
      NSData *encoded=[NSJSONSerialization dataWithJSONObject:names options:0 error:nil];
      fwrite(encoded.bytes,1,encoded.length,stdout); printf("\n"); return 0;
    }
    NSString *group=nil;
    NSMutableDictionary *channels=nil;
    for(NSString *candidate in @[@"PMP0",@"PMP"]){
      NSMutableDictionary *found=[(__bridge NSDictionary *)IOReportCopyChannelsInGroup((__bridge CFStringRef)candidate,CFSTR("DCS BW"),0,0,0) mutableCopy];
      if([found[@"IOReportChannels"] count]){ group=candidate; channels=found; break; }
    }
    if(!channels){
      NSDictionary *all=CFBridgingRelease(IOReportCopyAllChannels(0,0));
      for(NSDictionary *channel in all[@"IOReportChannels"]){
        CFDictionaryRef c=(__bridge CFDictionaryRef)channel;
        NSString *subgroup=(__bridge NSString *)IOReportChannelGetSubGroup(c);
        NSString *name=(__bridge NSString *)IOReportChannelGetChannelName(c);
        if([subgroup isEqualToString:@"DCS BW"]&&[name isEqualToString:@"AMCC RD"]){ group=[(__bridge NSString *)IOReportChannelGetGroup(c) copy]; break; }
      }
      if(group) channels=[(__bridge NSDictionary *)IOReportCopyChannelsInGroup((__bridge CFStringRef)group,CFSTR("DCS BW"),0,0,0) mutableCopy];
    }
    if(!channels){ printf("{\"up\":false}\n"); return 0; }
    CFMutableDictionaryRef subscribed=0;
    CFTypeRef subscription=IOReportCreateSubscription(0,(__bridge CFMutableDictionaryRef)channels,&subscribed,0,0);
    CFDictionaryRef a=IOReportCreateSamples(subscription,subscribed,0);
    do {
      usleep((useconds_t)ms*1000);
      CFDictionaryRef b=IOReportCreateSamples(subscription,subscribed,0);
      CFDictionaryRef delta=IOReportCreateSamplesDelta(a,b,0);
      emit_delta(delta,ms,group);
      CFRelease(delta);
      CFRelease(a);
      a=b;
    } while(streaming);
    CFRelease(a);
  }
}
