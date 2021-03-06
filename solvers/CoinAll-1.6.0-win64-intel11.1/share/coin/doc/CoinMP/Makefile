# Copyright (C) 2006 International Business Machines and others.
# All Rights Reserved.
# This file is distributed under the Common Public License.

# $Id: Makefile.in 726 2006-04-17 04:16:00Z andreasw $

##########################################################################
#    You can modify this example makefile to fit for your own program.   #
#    Usually, you only need to change the five CHANGEME entries below.   #
##########################################################################

EXNAME = example

# CHANGEME: This should be the name of your executable
EXE = $(EXNAME).exe

# CHANGEME: Here is the name of all object files corresponding to the source
#           code that you wrote in order to define the problem statement
OBJS =  $(EXNAME).obj

# CHANGEME: Additional libraries
ADDLIBS =

# CHANGEME: Additional flags for compilation (e.g., include flags)
ADDINCFLAGS =

# CHANGEME: Directory to the sources for the (example) problem definition
# files
SRCDIR = ../../../releases-1.6.0/CoinMP/examples
VPATH = ../../../releases-1.6.0/CoinMP/examples

##########################################################################
#  Usually, you don't have to change anything below.  Note that if you   #
#  change certain compiler options, you might have to recompile the      #
#  COIN package.                                                         #
##########################################################################

COIN_HAS_PKGCONFIG = #TRUE
COIN_CXX_IS_CL = TRUE
COIN_HAS_DATASTOCHASTIC = @COIN_HAS_DATASTOCHASTIC_TRUE@TRUE

# C Compiler command
CC = icl

# C Compiler options
CFLAGS = -MT -Ox -nologo -D_CRT_SECURE_NO_DEPRECATE -DNDEBUG    -DCOINMP_BUILD

# additional C Compiler options for linking
CLINKFLAGS = 

# Include directories (we use the CYGPATH_W variables to allow compilation with Windows compilers)
ifeq ($(COIN_HAS_PKGCONFIG), TRUE)
  INCL = `PKG_CONFIG_PATH=/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib/pkgconfig:/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/share/pkgconfig:  --cflags coinmp`
else
  INCL =  -I/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/include/coin  -I/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/include/coin/ThirdParty  -I/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/include/coin  -I/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/include/coin  -I/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/include/coin  -I/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/include/coin/ThirdParty  -I/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/include/coin  -I/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/include/coin  
endif
INCL += $(ADDINCFLAGS)

# Linker flags
ifeq ($(COIN_HAS_PKGCONFIG), TRUE)
  LIBS = `PKG_CONFIG_PATH=/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib/pkgconfig:/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/share/pkgconfig:  --libs coinmp`
else
  ifeq ($(COIN_CXX_IS_CL), TRUE)
    LIBS = -link -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libCoinMP.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libCbcSolver.lib libCbc.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libcoinasl.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libCgl.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libOsiClp.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libClp.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libcoinmumps.lib mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libOsi.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libCoinUtils.lib mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib   
  else
    LIBS = -L/usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib -lCoinMP   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libCbcSolver.lib libCbc.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libcoinasl.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libCgl.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libOsiClp.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libClp.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libcoinmumps.lib mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libOsi.lib   -libpath:`$(CYGPATH_W) /usr/home/svigerske/nbBuildDir/CoinAll/releases-1.6.0-default/lib` libCoinUtils.lib mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib   
  endif
endif

# The following is necessary under cygwin, if native compilers are used
CYGPATH_W = echo

all: $(EXE)

.SUFFIXES: .c .o .obj

$(EXE): $(OBJS)
	bla=;\
	for file in $(OBJS); do bla="$$bla `$(CYGPATH_W) $$file`"; done; \
	$(CC) $(CLINKFLAGS) $(CFLAGS) -o $@ $$bla $(LIBS) $(ADDLIBS)

clean:
	rm -rf $(EXE) $(OBJS)

.c.o:
	$(CC) $(CFLAGS) $(INCL) -c -o $@ `test -f '$<' || echo '$(SRCDIR)/'`$<

.c.obj:
	$(CC) $(CFLAGS) $(INCL) -c -o $@ `if test -f '$<'; then $(CYGPATH_W) '$<'; else $(CYGPATH_W) '$(SRCDIR)/$<'; fi`
