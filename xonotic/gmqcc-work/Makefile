






LTO ?= 0
ASAN ?= 0
UBSAN ?= 0
DEBUG ?= 0
UNUSED ?= 1
SRCDIR ?= ./



ifeq ($(OS),Windows_NT)
	GMQCC := gmqcc.exe
	QCVM := qcvm.exe
else
	GMQCC := gmqcc
	QCVM := qcvm
endif


CXX ?= clang++


OBJDIR := .build/objs
DEPDIR := .build/deps


GSRCS := ast.cpp
GSRCS += code.cpp
GSRCS += conout.cpp
GSRCS += fold.cpp
GSRCS += ftepp.cpp
GSRCS += intrin.cpp
GSRCS += ir.cpp
GSRCS += lexer.cpp
GSRCS += main.cpp
GSRCS += opts.cpp
GSRCS += parser.cpp
GSRCS += stat.cpp
GSRCS += utf8.cpp
GSRCS += util.cpp


QSRCS := exec.cpp
QSRCS += stat.cpp
QSRCS += util.cpp




CXXFLAGS := -Wall
CXXFLAGS += -Wextra
CXXFLAGS += -Wno-parentheses
CXXFLAGS += -Wno-class-memaccess
CXXFLAGS += -Wno-implicit-fallthrough
CXXFLAGS += -std=c++11


CXXFLAGS += -fno-exceptions
CXXFLAGS += -fno-rtti
CXXFLAGS += -fno-asynchronous-unwind-tables



ifeq ($(UNUSED),1)
	CXXFLAGS += -ffunction-sections
	CXXFLAGS += -fdata-sections
endif


ifeq ($(LTO),1)
	CXXFLAGS += -flto
endif

ifeq ($(DEBUG),1)

	CXXFLAGS += -fno-omit-frame-pointer


	CXXFLAGS += -O0


	CXXFLAGS += -g
else

	CXXFLAGS += -fno-stack-protector
	CXXFLAGS += -fno-stack-check


	ifeq ($(ASAN),1)
		CXXFLAGS += -fno-omit-frame-pointer
	else
		CXXFLAGS += -fomit-frame-pointer
	endif


	CXXFLAGS += -O3
endif


ifeq ($(ASAN),1)
	CXXFLAGS += -fsanitize=address
endif
ifeq ($(UBSAN),1)
	CXXFLAGS += -fsanitize=undefined
endif




DEPFLAGS := -MMD
DEPFLAGS += -MP




LDFLAGS :=


ifeq ($(UNUSED),1)
	LDFLAGS += -Wl,--gc-sections
endif


ifeq ($(LTO),1)
	LDFLAGS += -flto
endif


ifeq ($(ASAN),1)
	LDFLAGS += -fsanitize=address
endif
ifeq ($(UBSAN),1)
	LDFLAGS += -fsanitize=undefined
endif


ifneq (,$(findstring -g,$(CXXFLAGS)))
	STRIP := true
else
	STRIP := strip
endif

all: $(GMQCC) $(QCVM)


$(DEPDIR):
	@mkdir -p $(DEPDIR)
$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: %.cpp $(DEPDIR)/%.d | $(OBJDIR) $(DEPDIR)
	$(CXX) -MT $@ $(DEPFLAGS) -MF $(DEPDIR)/$*.Td $(CXXFLAGS) -c -o $@ $<
	@mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

$(GMQCC): $(filter %.o,$(GSRCS:%.cpp=$(OBJDIR)/%.o))
	$(CXX) $^ $(LDFLAGS) -o $@
	$(STRIP) $@

$(QCVM): $(filter %.o,$(QSRCS:%.cpp=$(OBJDIR)/%.o))
	$(CXX) $^ $(LDFLAGS) -o $@
	$(STRIP) $@

clean:
	rm -rf $(DEPDIR) $(OBJDIR)

.PHONY: clean $(DEPDIR) $(OBJDIR)


$(filter %.d,$(GSRCS:%.cpp=$(DEPDIR)/%.d)):
include $(wildcard $@)

$(filter %.d,$(QSRCS:%.cpp=$(DEPDIR)/%.d)):
include $(wildcard $@)
