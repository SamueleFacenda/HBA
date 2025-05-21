# nix comments
{
  description = "Example nix ros project";

  inputs = {
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/master";
    nixpkgs.follows = "nix-ros-overlay/nixpkgs";
  };

  nixConfig = {
    extra-substituters = [ "https://ros.cachix.org" ];
    extra-trusted-public-keys = [ "ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo=" ];
  };

  outputs = { self, nixpkgs, nix-ros-overlay }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.permittedInsecurePackages = [
          "freeimage-3.18.0-unstable-2024-04-18"
        ];
        overlays = [
          nix-ros-overlay.overlays.default
        ];
      };
    in {
      packages.${system}.hba = pkgs.rosPackages.noetic.callPackage ({
          buildRosPackage
        , catkin
        , roscpp
        , rospy
        , std-msgs
        , geometry-msgs
        , visualization-msgs
        , nav-msgs
        , sensor-msgs
        , rosbag
        , message-generation
        , tf
        , pcl
        , eigen
        , gtsam
        , message-runtime
        , pcl-conversions
        , tbb_2022_0
        }: 
        buildRosPackage {
          pname = "hba";
          version = "1.0.1";
          src = ./.;
          
          buildType = "catkin";
          nativeBuildInputs = [ catkin ];
          buildInputs = [
            pcl
            eigen
            tbb_2022_0.dev
          ];
          propagatedBuildInputs = [
            roscpp
            rospy
            std-msgs
            geometry-msgs
            visualization-msgs
            nav-msgs
            sensor-msgs
            rosbag
            message-generation
            tf
            message-runtime
            gtsam
            pcl-conversions
          ];
        }
      ) {};
    
      devShells.${system} = {
        default = pkgs.mkShell {
          buildInputs = [
            pkgs.glibcLocales
            pkgs.heaptrack
            pkgs.valgrind
            
            (pkgs.rosPackages.noetic.buildEnv {
              paths = with pkgs.rosPackages.noetic; [
                rosbash
                catkin
                rqt-graph
                teleop-twist-keyboard
                rqt-top
                rqt-reconfigure
                rqt-console
                rqt-logger-level
                rviz
                xacro
                self.packages.${system}.hba
              ];
            })
          ];

          ROS_HOSTNAME = "localhost";
          ROS_MASTER_URI = "http://localhost:11311";
          QT_QPA_PLATFORM = "xcb"; # qt on wayland
        };
      };
    };
}
