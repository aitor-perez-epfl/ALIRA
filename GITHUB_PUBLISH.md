# Step-by-Step Guide to Publish on GitHub

## Prerequisites
- A GitHub account (create one at https://github.com if you don't have one)
- Git installed on your system (check with `git --version`)

## Steps

### Step 1: Initialize Git Repository
Open your terminal in the project directory and run:

```bash
cd /Users/gaeberna/EPFL-local/2026-01-15-AL-LLM
git init
```

### Step 2: Add All Files to Git
Stage all your files for the first commit:

```bash
git add .
```

### Step 3: Create Your First Commit
Commit all the files:

```bash
git commit -m "Initial commit: ALIRA – Active Learning Iterative Retrieval Agent"
```

### Step 4: Create a New Repository on GitHub
1. Go to https://github.com and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Choose a repository name (e.g., "active-learning-llm" or "al-llm")
5. Add a description (optional): "ALIRA – Active Learning Iterative Retrieval Agent. Combines RAG with active learning to iteratively discover relevant documents from large corpora using LLM validation and classifier refinement."
6. Choose visibility (Public or Private)
7. **DO NOT** initialize with README, .gitignore, or license (you already have these)
8. Click "Create repository"

### Step 5: Connect Local Repository to GitHub
Connect to your repository at https://github.com/gaelbernard/ALIRA:

```bash
git remote add origin https://github.com/gaelbernard/ALIRA.git
git branch -M main
git push -u origin main
```

### Step 6: Verify
Go to your GitHub repository page and verify all files are uploaded correctly.

## Future Updates

When you make changes to your code, use these commands to update GitHub:

```bash
git add .
git commit -m "Description of your changes"
git push
```

## Troubleshooting

### If you get authentication errors:
GitHub no longer accepts passwords. You need to use:
- **Personal Access Token (PAT)**: Create one at https://github.com/settings/tokens
- Or use **SSH keys**: Set up SSH keys for GitHub

### If you need to change the remote URL:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### If you want to check your remote:
```bash
git remote -v
```
