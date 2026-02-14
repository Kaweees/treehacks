{
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python312 # Python 3.12
    uv # Python package manager
    nixfmt # Nix formatter
    just # Just runner
    cyclonedds
  ];

  shellHook = ''
    export CYCLONEDDS_HOME=${pkgs.cyclonedds}
    export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME:$CMAKE_PREFIX_PATH"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.wayland}/lib:${pkgs.libxkbcommon}/lib:${pkgs.libGL}/lib:${pkgs.libglvnd}/lib:${pkgs.mesa}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXcursor}/lib:${pkgs.xorg.libXrandr}/lib:${pkgs.xorg.libXi}/lib:$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH"
    export CFLAGS="-I$CYCLONEDDS_HOME/include/idlc $CFLAGS"
    export TMPDIR=/tmp
    just install
  '';
}