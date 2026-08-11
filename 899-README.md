# ECON 899 replication package template

This GitHub repo provides a template for ECON 899 students to create and share
their replication packages.

## Setup instructions

1. Create a *fork* of this repo in your own GitHub account.
2. Use Git or GitHub Desktop to *clone* your repo on your computer.

You are now ready to start adding and editing files to build your replication
package. Use Git/GitHub to track your changes and make regular commits.

## Template components

### SSDE README template

The `README.md` file in your final replication package should *not* be this
file.  Instead it should be a detailed set of replication instructions and
information based on the file `SSDE_readme.md` in this folder.

So one of your first commits should do the following:

1. Rename this file from `README.md` to `old_README.md`.
2. Make a copy of `SSDE_readme.md` and name it `README.md`

You can then start editing `README.md` to reflect your project.

### Folder structure

The repo already includes the basic folder structure for your replication package.

- All code goes in the `programs` folder
  - Code to clean and prepare the data for analysis goes in
    `programs/01_dataprep`.
  - Code to produce the tables and figures in the main text goes in
    `programs/02_analysis`.
  - Code to produce the tables and figures in the appendix goes in
    `programs/03_appendix`.
- All data goes in the `data` folder.
  - Raw data goes in `data/raw_data`.
  - Cleaned data for analysis goes in `data/data_for_analysis`.
- All results go in the `results` folder.

Each of these folders has a README file with additional details.

### Example programs

I have added some example Stata programs to the `programs` folder. You can use
them as a starting point for your own programs (if you are using Stata) or you
can create R/Python analogues.

## Advice

- Replicable research develops good habits, so use this structure for your data
  analysis from the very beginning.
- Make frequent commits with clear commit messages.
