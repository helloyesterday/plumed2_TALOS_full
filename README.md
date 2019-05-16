# plumed2_TALOS_full
The modified version of PLUMED2 included targeted adversarial learning optimizied sampling (TALOS)

This version is based on the offical verision of PLUMED 2.5 with VES code. The module of TALOS need the DyNet C++ library for deep learning (http://dynet.io/). To use the function of TALOS, you need to enable all the modules at configuration:

  ./configure --enable-modules=all 
