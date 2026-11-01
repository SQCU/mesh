#pragma once
#include "mesh-host.h"
#import <Foundation/Foundation.h>
#import <objc/message.h>

/* The request and its pages have already been bound. This is the common ABI
   for the native entry and source-specialized entries with page effects. */
typedef struct {
  void *client, *model, *options, *request;
  BOOL (*evaluate)(id, SEL, id, id, id, unsigned, NSError **);
  unsigned qos;
  MeshHostCompletion complete;
  void *completion_context;
  const void *arguments;
  void *owner;
} MeshANEEntry;
typedef void (*MeshANEConfigure)(MeshANEEntry *);

static inline void MeshANEEvaluateRequest(MeshANEEntry *entry) {
  NSError *error=nil;
  if(!entry->evaluate((__bridge id)entry->client,@selector(evaluateWithModel:options:request:qos:error:),
      (__bridge id)entry->model,(__bridge id)entry->options,(__bridge id)entry->request,entry->qos,&error))
    entry->complete(entry->completion_context,error.localizedDescription.UTF8String?:"direct ANE submission failed");
}
