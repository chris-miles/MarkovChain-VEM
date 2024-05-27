# MarkovChain-VEM

Code repository for "Practical Markov chain clustering: Fast, automatic
determination of the number of clusters" by RJW and CEM.

Jupyter notebooks in main folder reproduce figures from the paper.

The main code to run the VEM algorithm can be found in [/code/clustering.py]([/code/clustering.py])

Any issues or questions can be directed to chris.miles@uci.edu. 

### Figure 1

Run [example_fig.ipynb](example_fig.ipynb) generates both subpanels of the figure as separate files. 

### Figure 2

Run [sweep_accuracyNT_fig.ipynb](sweep_accuracyNT_fig.ipynb) generates both subpanels of the figure as separate files. 

### Figure 3

Run [EM_localmin_fig.ipynb](EM_localmin_fig.ipynb) generates both subpanels of the figure as separate files. 

### Figure 4

This figure and the next depend on the `Last.fm` dataset. We have included in the data folder `Lastfm-ArtistTags2007` but the data folder also requires `lastfm-dataset-1K` which is considerably larger (~ 1 GB) and not uploaded to Github, but can be found here: http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz. If this link dies, please email me and I’ll send it to you.

The Last.fm data processing is copied from  https://github.com/hetankevin/mdpmix so the performance comparison is as direct as possible. 

Run [lastfm_top10_fig.ipynb](lastfm_top10_fig.ipynb) after both artist tags and user data have been added to the data folder. This will generate both panels as individual figures.  

### Figure 5

See previous figure. Run [lastfm_clusterusers.ipynb](lastfm_clusterusers.ipynb) to generate a single figure with both panels.

### Figure 6

The Ultrarunner dataset is available from http://maths.ucd.ie/~brendan/data/24H.xlsx, as referenced in the [Roick et al paper](https://doi.org/10.1007/s11634-020-00395-7). We have included it in the `data/` folder.

Run [ultrarunners_fig.ipynb](ultrarunners_fig.ipynb) to generate the full figure.

### Figure 7

The simulations in the Jupyter notebook use the [GillesPy2](https://github.com/StochSS/GillesPy2) and could be optimized further by using the option to compile to `C`. 

Run [gene_examples.ipynb](gene_examples.ipynb) to generate the full figure. 

### Figure 8

Run [gene_clustering_acc_fig.ipynb](gene_clustering_acc_fig.ipynb) to generate the full figure. 