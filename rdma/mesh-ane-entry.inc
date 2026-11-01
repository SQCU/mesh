/* One entry recipe. Compilation inserts effects here, before request binding.
   Empty effects are the same recipe for a wholly local provider invocation. */
#include "mesh-ane-entry.h"
#ifndef MESH_ANE_ENTER
#define MESH_ANE_ENTER(arguments)
#define MESH_ANE_LEAVE(arguments)
#endif
#ifdef __cplusplus
extern "C" {
#endif
void mesh_ane_submit(void *context) {
  MeshANEEntry *entry=(MeshANEEntry *)context;
  MESH_ANE_ENTER(entry->arguments);
  MeshANEEvaluateRequest(entry);
}
void mesh_ane_configure(MeshANEEntry *entry) {
  void (^completed)(BOOL,NSError *)=^(BOOL success,NSError *error) {
    if(success) { MESH_ANE_LEAVE(entry->arguments); }
    entry->complete(entry->completion_context,
      success?NULL:(error.localizedDescription.UTF8String?:"direct ANE execution failed"));
  };
  ((void(*)(id,SEL,id))objc_msgSend)((__bridge id)entry->request,@selector(setCompletionHandler:),completed);
}
#ifdef __cplusplus
}
#endif
