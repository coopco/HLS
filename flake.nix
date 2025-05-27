{
  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url  = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs      = import nixpkgs { inherit system; };
        python    = pkgs.python311;
        pyPkgs    = pkgs.python311Packages;
      in {
        devShell = pkgs.mkShell {
          name = "hls-devshell";

          buildInputs = with pkgs; [


    #pkgs.zlib
    #pkgs.suitesparse  # Dependency for scikit-sparse
    #pkgs.R
    #pkgs.rPackages.glmnet
    #pkgs.python311Packages.rpy2
          ];

          # Preserve your LD_LIBRARY_PATH logic
          shellHook = ''
            export LD_LIBRARY_PATH="${lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
          '';
        };
      });
}

{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    {
      devShells.x86_64-linux =
        let
          pkgs = nixpkgs.legacyPackages.x86_64-linux;
        in
        {
          default = pkgs.mkShell rec {
            packages = [
              pkgs.python311
              pkgs.python311Packages.cython
              pkgs.python311Packages.setuptools
              pkgs.gcc
              pkgs.gnumake
              pkgs.uv
              pkgs.zlib
              pkgs.suitesparse
              pkgs.R
              pkgs.rPackages.glmnet
              pks.python311Packages.setuptools
              pks.python311Packages.pip
              pks.python311Packages.rpy2
              pks.python311Packages.debugpy
            ];
            shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.libsdf.makeLibraryPath  packages}:$LD_LIBRARY_PATH"
              export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
            '';
          };
        };
    };
}
