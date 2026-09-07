"""
Master Curriculum and Project Generator for Tensorbox.
Runs all generators to produce 37 Jupyter Notebooks and 10 Enterprise Projects.
"""

from .tracks_01_to_03 import build_tracks_01_to_03
from .tracks_04_to_06 import build_tracks_04_to_06
from .tracks_07_to_10 import build_tracks_07_to_10
from .tracks_11_to_14 import build_tracks_11_to_14
from .projects_01_to_05 import build_projects_01_to_05
from .projects_06_to_10 import build_projects_06_to_10

def main():
    print("==================================================")
    print("🚀 TENSORBOX CURRICULUM & ENTERPRISE PROJECT BUILDER")
    print("==================================================")
    
    print("\n--- Generating Tracks 01 to 03 (9 Notebooks) ---")
    build_tracks_01_to_03()
    
    print("\n--- Generating Tracks 04 to 06 (7 Notebooks) ---")
    build_tracks_04_to_06()
    
    print("\n--- Generating Tracks 07 to 10 (8 Notebooks) ---")
    build_tracks_07_to_10()
    
    print("\n--- Generating Tracks 11 to 14 (13 Notebooks) ---")
    build_tracks_11_to_14()
    
    print("\n--- Generating Flagship Projects 01 to 05 ---")
    build_projects_01_to_05()
    
    print("\n--- Generating Flagship Projects 06 to 10 ---")
    build_projects_06_to_10()
    
    print("\n==================================================")
    print("🎉 ALL 37 NOTEBOOKS & 10 PROJECTS SUCCESSFULLY GENERATED!")
    print("==================================================")

if __name__ == "__main__":
    main()
