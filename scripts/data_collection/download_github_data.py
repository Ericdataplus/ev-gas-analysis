"""
GitHub Data Collector

Fetches relevant datasets and resources from GitHub repositories
for EV vs Gas analysis.

Uses the GitHub API to:
- Find relevant repositories
- Download data files directly
- Clone useful analysis repos for reference
"""

import requests
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "github"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# GitHub API base URL
GITHUB_API = "https://api.github.com"

# Relevant repositories with data
GITHUB_REPOS = {
    "open_ev_data": {
        "repo": "KilowattApp/open-ev-data",
        "description": "Open EV specifications dataset",
        "files_to_download": ["data/ev-data.json"],
    },
    "ev_charging_caltech": {
        "repo": "JuliaSokolova/electric-vehicle-charging",
        "description": "EV charging patterns analysis",
        "clone": True,  # Clone the whole repo for analysis code
    },
    "ev_infrastructure": {
        "repo": "Cody-Lange/Exploring-Electric-Vehicle-Infrastructure",
        "description": "EV infrastructure analysis in US",
        "clone": True,
    },
}

# Direct data sources (raw files on GitHub)
GITHUB_RAW_DATA = {
    "nrel_alt_fuel_stations": {
        "url": "https://afdc.energy.gov/api/alt-fuel-stations/v1.json?api_key=DEMO_KEY&fuel_type=ELEC&country=US&status=E&limit=10000",
        "filename": "nrel_ev_stations.json",
        "description": "NREL Alternative Fuel Stations API (EV only)",
    },
}


def search_github_repos(query: str, max_results: int = 10) -> list:
    """Search GitHub for relevant repositories."""
    print(f"Searching GitHub for: {query}")
    
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": max_results,
    }
    
    try:
        response = requests.get(f"{GITHUB_API}/search/repositories", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        repos = []
        for item in data.get("items", []):
            repos.append({
                "name": item["name"],
                "full_name": item["full_name"],
                "description": item.get("description", "No description"),
                "stars": item["stargazers_count"],
                "url": item["html_url"],
            })
        
        return repos
        
    except Exception as e:
        print(f"Error searching GitHub: {e}")
        return []


def download_file_from_github(repo: str, file_path: str, save_path: Path) -> bool:
    """Download a specific file from a GitHub repository."""
    raw_url = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"
    
    print(f"  Downloading: {file_path}")
    
    try:
        response = requests.get(raw_url, timeout=60)
        
        # Try master branch if main fails
        if response.status_code == 404:
            raw_url = f"https://raw.githubusercontent.com/{repo}/master/{file_path}"
            response = requests.get(raw_url, timeout=60)
        
        response.raise_for_status()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(response.content)
        print(f"  ✓ Saved to {save_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def clone_repo(repo: str, target_dir: Path) -> bool:
    """Clone a GitHub repository."""
    repo_url = f"https://github.com/{repo}.git"
    
    print(f"  Cloning: {repo}")
    
    try:
        if target_dir.exists():
            print(f"  → Already exists, skipping")
            return True
        
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode == 0:
            print(f"  ✓ Cloned to {target_dir}")
            return True
        else:
            print(f"  ✗ Clone failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def download_external_data():
    """Download data from external APIs and sources."""
    print("\n" + "-" * 50)
    print("DOWNLOADING EXTERNAL API DATA")
    print("-" * 50)
    
    for key, info in GITHUB_RAW_DATA.items():
        url = info["url"]
        filename = info["filename"]
        description = info["description"]
        save_path = DATA_DIR / filename
        
        print(f"\n{description}")
        print(f"  Source: {url[:80]}...")
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            save_path.write_bytes(response.content)
            print(f"  ✓ Saved to {save_path}")
            
            # Parse and show summary
            if filename.endswith(".json"):
                data = response.json()
                if isinstance(data, dict):
                    if "fuel_stations" in data:
                        print(f"  → Contains {len(data['fuel_stations'])} fuel stations")
                    else:
                        print(f"  → Keys: {list(data.keys())[:5]}")
                        
        except Exception as e:
            print(f"  ✗ Error: {e}")


def main():
    """Main function to collect GitHub data."""
    print("=" * 60)
    print("GITHUB DATA COLLECTOR")
    print("EV vs Gas Environmental Analysis")
    print("=" * 60)
    
    # Search for additional relevant repos
    print("\n" + "-" * 50)
    print("SEARCHING FOR RELEVANT REPOSITORIES")
    print("-" * 50)
    
    search_queries = [
        "electric vehicle data analysis",
        "EV charging infrastructure",
        "gas station locations USA",
    ]
    
    all_repos = []
    for query in search_queries:
        repos = search_github_repos(query, max_results=5)
        all_repos.extend(repos)
        
    # Print unique repos
    seen = set()
    print("\nTop repositories found:")
    for repo in all_repos:
        if repo["full_name"] not in seen:
            seen.add(repo["full_name"])
            print(f"  ★ {repo['stars']:,} - {repo['full_name']}")
            print(f"       {repo['description'][:60] if repo['description'] else 'No description'}...")
    
    # Download from known repos
    print("\n" + "-" * 50)
    print("DOWNLOADING FROM KNOWN REPOSITORIES")
    print("-" * 50)
    
    for key, info in GITHUB_REPOS.items():
        repo = info["repo"]
        description = info["description"]
        
        print(f"\n{description}")
        print(f"  Repo: {repo}")
        
        if info.get("clone"):
            target_dir = DATA_DIR / key
            clone_repo(repo, target_dir)
        else:
            for file_path in info.get("files_to_download", []):
                save_path = DATA_DIR / key / Path(file_path).name
                download_file_from_github(repo, file_path, save_path)
    
    # Download external data
    download_external_data()
    
    print("\n" + "=" * 60)
    print(f"DATA COLLECTION COMPLETE")
    print(f"Data saved to: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
