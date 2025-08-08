# Code base for the NOSTR paper

## File description
Each directory have to be used for what it mean:

- computation: all files that possess the functions to create new computations of datas
- sampling: all files that, if executed, creates the differents computations according to parameters inside each one of them
- plottig: all files too create the differents plots
- data, rdata, bdata and idata: all images and graphs created by the programs

To execute the files correctly, you can execute the following files in the directory `sampling`:
- big
- real
- simple

---

## Informations over execution
All libraries used in this codebase are in the `requirements.txt` file, you should start by running the command:
```bash
pip install -r requirements.txt
```
You can execute the files by using variations of the command
```bash
python sampling/filename.py
```

**BE CAREFULL**, the files are made to execute a total of 1000 sample over 100 files, if you want to change it, change it in the code directly.

### Additional notes for the file real.py

This file have multiple way to be compiled, because of the presence of multiple _real files_. Because of this, it's required to decide before executing the file the graph you want to run, by commenting or not the lines.

**BE CAREFULL**, the twitter file isn't runable, some changes needs to be done over the algorithm because of the presence/absence of specific nodes in the graph.
**BE CAREFULL**, I never had the patience to do the computation for the real files, so it's possible that their saving address is the same for everyone of them, be carefull to either change manually their names or to change the code so that it can be done by itself.


### How to verify the work
If you want to be sure that all the datas are _enough_ for what you need, a function is executed in the end of each sampling files to verify the correlation of each files created. Don't hesitate to note the informations

## How the plotting work
To create all the plots, just run all ipynb files in the plotting directory. If you are working on the paper, the graphs validated for the project are the one in `all.ipynb`.
