#pragma once
/* One source ABI for a retained numerical site's empty effects. Resource
   analysis uses this entry too; final page-effect emission specializes the
   same hooks and ordinary metadata operand before execution binding. */
#define MESH_PAGE_SITES_ABI(pointer) \
  "#ifndef MESH_PAGE_SITE\n" \
  "template<class Site> struct mesh_page_site {\n" \
  "static inline void enter(" pointer ",unsigned long,unsigned int) {}\n" \
  "static inline void leave(" pointer ",unsigned long,unsigned int) {} };\n" \
  "#define MESH_PAGE_SITE(name) struct mesh_page_site_##name;\n" \
  "#define MESH_PAGE_ENTER(name,metadata,cell,lane) mesh_page_site<mesh_page_site_##name>::enter(metadata,cell,lane)\n" \
  "#define MESH_PAGE_LEAVE(name,metadata,cell,lane) mesh_page_site<mesh_page_site_##name>::leave(metadata,cell,lane)\n" \
  "#define MESH_PAGE_OPERAND ((" pointer ")nullptr)\n" \
  "#endif\n"
#define MESH_PAGE_SITES_SOURCE MESH_PAGE_SITES_ABI("device unsigned char *")
#define MESH_PAGE_HOST_SITES_SOURCE MESH_PAGE_SITES_ABI("unsigned char *")
