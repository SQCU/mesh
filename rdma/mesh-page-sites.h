#pragma once
/* One source ABI for a retained numerical site's empty effects. Resource
   analysis uses this entry too; final page-effect emission specializes the
   same hooks and ordinary metadata operand before execution binding. */
#define MESH_PAGE_SITES_SOURCE \
  "#ifndef MESH_PAGE_SITE\n" \
  "template<class Site> struct mesh_page_site {\n" \
  "static inline void enter(device unsigned char *,unsigned long,unsigned int) {}\n" \
  "static inline void leave(device unsigned char *,unsigned long,unsigned int) {} };\n" \
  "#define MESH_PAGE_SITE(name) struct mesh_page_site_##name;\n" \
  "#define MESH_PAGE_ENTER(name,metadata,cell,lane) mesh_page_site<mesh_page_site_##name>::enter(metadata,cell,lane)\n" \
  "#define MESH_PAGE_LEAVE(name,metadata,cell,lane) mesh_page_site<mesh_page_site_##name>::leave(metadata,cell,lane)\n" \
  "#define MESH_PAGE_OPERAND ((device uchar *)nullptr)\n" \
  "#endif\n"
