{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell rec {

  buildInputs = [
    pkgs.python3
    pkgs.python311Packages.cython
    pkgs.python311Packages.pip
    pkgs.python311Packages.debugpy
    pkgs.python311Packages.setuptools
    pkgs.poetry
    pkgs.zlib
    pkgs.suitesparse  # Dependency for scikit-sparse

    pkgs.R
    pkgs.rPackages.glmnet
    pkgs.python311Packages.rpy2
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
  '';
}
