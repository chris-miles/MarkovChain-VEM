# MarkovChain-VEM

Code repository for "Practical Markov chain clustering: Fast, automatic determination of the number of clusters" by RJW and CEM.

The core code to run the VEM algorithm can be found in [code/clustering.py](code/clustering.py) with function `doVEM` that does a single instance of VEM. There are also wrappers `doVEMmulti` and `doVEMmultiPar` that run several (parallelized) initiations and outputs the run corresponding to the highest ELBO. 

Any issues or questions can be directed to chris.miles@uci.edu or rwebber@ucsd.edu.

---

Jupyter notebooks in main repo folder reproduce figures from the paper. Postprocessing (slight font tweaks, combining panels) was done in Adobe Illustrator. 

### Figure 1: example clustering

The notebook [example.ipynb](example.ipynb) generates both subpanels of the figure as separate files. 

### Figure 2: classification accuracy sweeps against $N$, $T$ 

The notebook [sweep_accuracyNT.ipynb](sweep_accuracyNT.ipynb) generates both subpanels of the figure as separate files. 

### Figure 3: local minima in VEM 

The notebook [EM_localmin.ipynb](EM_localmin.ipynb) generates the figure. 

### Figure 4: Last.fm top 10 user identification

This figure and the next depend on the Last.fm  dataset. We have included in the data folder `Lastfm-ArtistTags2007` but the data folder also requires `lastfm-dataset-1K` which is considerably larger (>2 GB) and not uploaded to Github, but can be found here: http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz. If this link dies, please email me and I’ll send it to you.

The Last.fm data processing is copied from  https://github.com/hetankevin/mdpmix so the performance comparison is as direct as possible. 

Run [lastfm_top10_sweep.ipynb](lastfm_top10sweep.ipynb) after both artist tags and user data have been added to the data folder. This will generate the sweep of accuracy against trajectory length $T$. 

The notebook [lastfm_confusion.ipynb](lastfm_confusion.ipynb) generates the confusion matrix for $T=250$ shown in panel b.

We also included [lastfm_top10_sweep_keepgenre.ipynb](lastfm_top10_sweep_keepgenre.ipynb), a version of the similarly named notebook where we do not discard same-genre transitions. However, no impact on accuracy is seen.

### Figure 5: Last.fm number of user clusters

See previous figure. Run [lastfm_clusterusers.ipynb](lastfm_clusterusers.ipynb) to generate a single figure with both panels.

We include in this repo but not the paper a test where we do not discard same-genre transitions, shown in [lastfm_clusterusers_keepgenre.ipynb](lastfm_clusterusers_keepgenre.ipynb). 

### Figure 6: Ultrarunner clustering

The Ultrarunner dataset is available from http://maths.ucd.ie/~brendan/data/24H.xlsx, as referenced in the [Roick et al paper](https://doi.org/10.1007/s11634-020-00395-7). We have included it in the `data/` folder.

Run [ultrarunners.ipynb](ultrarunners.ipynb) to generate the full figure.

### Figure 7: MISA gene circuit schematic

The simulations in the Jupyter notebook use the [GillesPy2](https://github.com/StochSS/GillesPy2) and could be optimized further by using the option to compile to `C`. 

The Markov State spectral clustering utilizes some speedups and code from RPCholesky:
https://github.com/eepperly/Randomly-Pivoted-Cholesky

Run [gene_examples.ipynb](gene_examples.ipynb) to generate the full figure. 

### Figure 8: MISA gene circuit clustering accuracy

Run [gene_clustering_acc.ipynb](gene_clustering_acc.ipynb) to generate the full figure. 