# NMFLibrary
MATLAB library for non-negative matrix factorization (NMF)

Authors: [Hiroyuki Kasai](http://kasai.comm.waseda.ac.jp/kasai/)

Last page update: Oct. 14, 2020

Latest library version: 1.8.1 (see Release notes for more info)

<br />

Introduction
----------
The NMFLibrary is a **pure-Matlab** library of a collection of algorithms of **non-negative matrix factorization (NMF)**. 

<br />

## <a name="supp_solver"> List of the algorithms available in NMFLibrary </a>

- Base NMF
    - **MU** (multiplicative updates)
        - MU
            - D.D. Lee and H. S. Seung, "[Algorithms for non-negative matrix factorization](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf)," NIPS 2000. (for Euclidean distance and [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence) (KL))
            - A.Cichocki, S.Amari, R.Zdunek, R.Kompass, G.Hori, and Z.He, "[Extended SMART algorithms for non-negative matrix factorization](https://link.springer.com/chapter/10.1007/11785231_58)," Artificial Intelligence and Soft Computing, 2006. (for alpha divergence and beta divergence)
        - Modified MU
            - C.-J. Lin, "[On the convergence of multiplicative update algorithms for nonnegative matrix factorization](http://ieeexplore.ieee.org/document/4359171/)," IEEE Trans. Neural Netw. vol.18, no.6, pp.1589-1596, 2007. 
        - Acceralated MU
            - N. Gillis and F. Glineur, "[Accelerated multiplicative updates and hierarchical ALS algorithms for nonnegative matrix factorization](https://arxiv.org/pdf/1107.5194.pdf)," Neural Computation, vol.24, no.4, pp. 1085-1105, 2012. 

    - **PGD** (projected gradient descent)
        - PGD
        - Direct PGD
            - C.-J. Lin, "[Projected gradient methods for nonnegative matrix factorization](https://www.csie.ntu.edu.tw/~cjlin/papers/pgradnmf.pdf)," Neural Computation, vol.19, no.10, pp.2756-2779, 2007.

    - **ALS** (alternative least squares)
        - ALS
        - Hierarchical ALS (HALS)
            - A. Cichocki and P. Anh-Huy, "[Fast local algorithms for large scale nonnegative matrix and tensor factorizations](http://www.bsp.brain.riken.jp/publications/2009/Cichocki-Phan-IEICE_col.pdf)," IEICE Trans. on Fundamentals of Electronics, Communications and Computer Sciences, vol.92, no.3, pp. 708-721, 2009.
        - Acceralated Hierarchical ALS
            - N. Gillis and F. Glineur, "[Accelerated multiplicative updates and hierarchical ALS algorithms for nonnegative matrix factorization](https://arxiv.org/pdf/1107.5194.pdf)," Neural Computation, vol.24, no.4, pp. 1085-1105, 2012. 

    - **ANLS** (alternative non-negative least squares)
        - ASGROUP (ANLS with Active Set Method and Column Grouping)
        - ASGIVENS (ANLS with Active Set Method and Givens Updating)
        - BPP (ANLS with Block Principal Pivoting Method)
            - J. Kim, Y. He, and H. Park, "[Algorithms for nonnegative matrix and tensor factorizations: A unified view based on block coordinate descent framework](https://link.springer.com/article/10.1007/s10898-013-0035-4)," Journal of Global Optimization, 58(2), pp. 285-319, 2014.
            - J. Kim and H. Park, "[Fast nonnegative matrix factorization: An active-set-like method and comparisons](http://epubs.siam.org/doi/abs/10.1137/110821172)," SIAM Journal on Scientific Computing (SISC), 33(6), pp. 3261-3281, 2011.

- Variant

    - **GNMF** (Graph Regularized NMF)
        - D. Cai, X. He, X. Wu, and J. Han, "[Non-negative Matrix Factorization on Manifold](https://ieeexplore.ieee.org/document/4781101/)," Proc. 2008 Int. Conf. on Data Mining (ICDM), 2008. 
        - D. Cai, X. He, J. Han and T. Huang, "[Graph Regularized Non-negative Matrix Factorization for Data Representation](https://ieeexplore.ieee.org/document/5674058/)," IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol.33, No.8, pp.1548-1560, 2011. 

    - **Semi-NMF** 
        - C.H.Q. Ding, T. Li, M. I. Jordan, "[Convex and Semi-Nonnegative Matrix Factorizations](https://ieeexplore.ieee.org/document/4685898/)," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.32, no.1, 2010. 

    - **NeNMF** (NMF with Sinkhorn Distance)
        - N. Guan, D. Tao, Z. Luo, and B. Yuan, "[NeNMF: An Optimal Gradient Method for Non-negative Matrix Factorization](https://ieeexplore.ieee.org/document/6166359)", IEEE Transactions on Signal Processing, Vol. 60, No. 6, pp. 2882-2898, Jun. 2012.

    - **SDNMF** (NMF with Sinkhorn Distance)
        - W. Qian, B. Hong, D. Cai, X. He, and X. Li, "[Non-negative matrix factorization with sinkhorn distance](https://pdfs.semanticscholar.org/42b2/ec6e4453dc033d37b9fcb53a1313c018fa23.pdf)", IJCAI, pp.1960-1966, 2016.


    
- Robust NMF

    - N. Guan, D. Tao, Z. Luo, and B. Yuan, "[Online nonnegative matrix factorization with robust stochastic approximation](https://ieeexplore.ieee.org/document/6203594/)," IEEE Trans. Newral Netw. Learn. Syst., 2012.

- Sparse

    - **sparseMU** (Sparse multiplicative upates (MU))
        - J. Eggert and E. Korner, "[Sparse coding and NMF](https://ieeexplore.ieee.org/document/1381036/)", IEEE International Joint Conference on Neural Networks, 2004.
        - M. Schmidt, J. Larsen, and F. Hsiao, "[Wind noise reduction using non-negative sparse coding](https://ieeexplore.ieee.org/document/4414345/)", IEEE Workshop on Machine Learning for Signal Processing (MLSP), 2007.

    - **sparseNMF** (Sparse NMF)

    - **NMFsc** (NMF with sparseness constraints)
        - Patrik O. Hoyer, "[Non-negative matrix factorization with sparseness constraints](http://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf)," Journal of Machine Learning Research (JMLR), vol.5, pp.1457-1469, 2004.

    - **nsNMF** (Nonsmooth NMF)
        - A. Pascual-Montano, J. M. Carazo, K. Kochi, D. Lehmann, and R. D. Pascual-Marqui, "[Nonsmooth Nonnegative Matrix Factorization (nsNMF)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1580485)," IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), vol.28, no.3, pp.403-415, 2006. 

    - **fnsNMF** (Fast nonsmooth NMF)
        - Z. Yang, Y. Zhang, W. Yan, Y. Xiang, and S. Xie, "[A fast non-smooth nonnegative matrix factorization for learning sparse representation](https://ieeexplore.ieee.org/document/7559804/)," IEEE Access, vol.4, pp.5161-5168, 2016.

    - **NMF-HALS-SO** (Hierarchical ALS with soft orthogonal constraint)
        - M. Shiga, K. Tatsumi, S. Muto, K. Tsuda, Y. Yamamoto, T. Mori, and T. Tanji, "[Sparse modeling of EELS and EDX spectral imaging data by nonnegative matrix factorization](https://www.sciencedirect.com/science/article/pii/S0304399116301267)", Ultramicroscopy, Vol.170, p.43-59, 2016.
   
- Orthgotonal

    - **DTPP** (Orthgotonal multiplicative upates (MU))
        - C. Ding, T. Li, W. Peng, and H. Park, "Orthogonal nonnegative matrix t-factorizations for clustering", 12th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD), 2006.

    - **orthMU** (Orthgotonal multiplicative upates (MU))
        - S. Choi, "[Algorithms for orthogonal nonnegative matrix factorization](https://ieeexplore.ieee.org/document/4634046/)", IEEE International Joint Conference on Neural Networks, 2008.

    - **OrthNMF**
        - F. Pompilia, N. Gillis, P.-A. Absil, and F. Glineur, "[Two algorithms for orthogonal nonnegative matrix factorization with application to clustering](https://www.sciencedirect.com/science/article/pii/S0925231214004068)," Neurocomputing, vol.141, no.2, pp.15-25, 2014.

- Symmetric

    - **SymmANLS** (Symmetric ANLS)
        - D. Kuang, C. Ding, H. Park, "[Symmetric Nonnegative Matrix Factorization for Graph Clustering](https://epubs.siam.org/doi/abs/10.1137/1.9781611972825.10?mobileUi=0)," The 12th SIAM International Conference on Data Mining (SDM'12), pp.106-117, 2012.

        - D. Kuang, S. Yun, H. Park, "[SymNMF Nonnegative low-rank approximation of a similarity matrix for graph clustering](https://link.springer.com/article/10.1007/s10898-014-0247-2)," Journal of Global Optimization, vol.62, no.3, pp.545-574, 2015.

        - Z. Zhu, X. Li, K. Liu, Q. Li, "[Dropping Symmetry for Fast Symmetric Nonnegative Matrix Factorization](https://papers.nips.cc/paper/7762-dropping-symmetry-for-fast-symmetric-nonnegative-matrix-factorization)," NIPS, 2018.

    - **SymmHALS** (Symmetric HALS)
        - Z. Zhu, X. Li, K. Liu, Q. Li, "Dropping Symmetry for Fast Symmetric Nonnegative Matrix Factorization," NIPS, 2018.

    - **SymmNewton** (Symmetric Newton)

- Online/stochstic NMF

    - **INMF** (Incremental NMF) and **ONMF** (Online NMF)
        - S. S. Bucak and B. Gunsel, "[Incremental Subspace Learning via Non-negative Matrix Factorization](https://www.sciencedirect.com/science/article/pii/S0031320308003725)," Pattern Recognition, 2009.

    - **SPG** (Stochastic projected gradient descent)

    - **RONMF** (Robust online NMF)
        - R. Zhao and Y. F. Tan, "[Online nonnegative matrix factorization with outliers](https://ieeexplore.ieee.org/document/7676413/)," IEEE ICASSP2016, 2016.
        - N. Guan, D. Tao, Z. Luo, and B. Yuan, "[Online nonnegative matrix factorization with robust stochastic approximation](https://ieeexplore.ieee.org/document/6203594/)," IEEE Trans. Newral Netw. Learn. Syst., 2012.

    - **SAGA-MU-NMF** (SAGA multiplicative updates)
        - R. Serizel, S. Essid and G.Richard, "[Mini-batch stochastic approaches for accelerated multiplicative updates in nonnegative matrix factorisation with beta-divergence](https://ieeexplore.ieee.org/document/7738818/),", IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP), 2016.

    - **SMU** (Stochastic multiplicative updates) and **SVRMU** (Stochastic variance reduced multiplicative updates)
        - H. Kasai, "[Stochastic variance reduced multiplicative update for nonnegative matrix factorization](https://arxiv.org/abs/1710.10781)," IEEE ICASSP2018, 2018.

- Probabilistic NMF

    - **PNMF-GIBBS** (Gibbs sampler for non-negative matrix factorisation, with ARD.)
        - M.N. Schmidt, O. Winther, L.K. Hansen, "[Bayesian non-negative matrix factorization](https://link.springer.com/chapter/10.1007/978-3-642-00599-2_68)," International Conference on Independent Component Analysis and Signal Separation, Springer Lecture Notes in Computer Science, Vol. 5441, 2009.

        - T. Brouwer, P. Lio, "[Bayesian Hybrid Matrix Factorisation for Data Integration](http://proceedings.mlr.press/v54/brouwer17a/brouwer17a.pdf)," 20th International Conference on Artificial Intelligence and Statistics (AISTATS), 2017.
        
    - **PNMF-VB** (Variational Bayesian inference for non-negative matrix factorisation, with ARD)
        - T. Brouwer, J. Frellsen. P. Lio, "[Comparative Study of Inference Methods for Bayesian Nonnegative Matrix Factorisation](https://link.springer.com/chapter/10.1007/978-3-319-71249-9_31)," ECML PKDD 2017, 2017.


<br />

## Algorithm configurations


|Category|Name in example codes| function | `options.alg` | other `options` |
|---|---|---|---|---|
|Base|MU-EUC|`nmf_mu`|`mu`|`metric='EUC'`|
||MU-KL|`nmf_mu`|`mu`|`metric='KL'`|
||MU-ALPHA|`nmf_mu`|`mu`|`metric='ALPHA-D'`|
||MU-BETA|`nmf_mu`|`mu`|`metric='BETA-D'`|
||Modified MU|`nmf_mu`|`mod_mu`||
||Acceralated MU|`nmf_mu`|`acc_mu`||
||PGD|`nmf_pgd`|`pgd`||
||Direct PGD|`nmf_pgd`|`direct_pgd`||
||ALS|`nmf_als`|`als`||
||Hierarchical ALS|`nmf_als`|`hals_mu`||
||Acceralated hierarchical ALS|`nmf_als`|`acc_hals_mu`||
||ASGROUP|`nmf_anls`|`anls_asgroup`||
||ASGIVENS|`nmf_anls`|`anls_asgivens`||
||BPP|`nmf_anls`|`anls_bpp`||
|Variant|Semi-NMF|`semi_nmf`|||
||NeNMF|`nenmf`|||
||GNMF|`GNMF`|||
||SDNMF|`SDNMF`|||
|Sparse|sparseMU-EUC|`nmf_sparse_mu`||`metric='EUC'`|
||sparseMU-KL|`nmf_sparse_mu`||`metric='KL'`|
||sparseNMF|`sparse_nmf`|||
||NMFsc|`nmf_sc`|||
||nsNMF|`ns_nmf`|||
||fnsNMF|`ns_nmf`||`metric='EUC'`, `update_alg='apg'`|
|Orthogonal|DTPP|`nmf_dtpp`|||
||orthMU|`nmf_orth_mu`|||
||OrthNMF||||
||NMF-HALS-SO||||
|Symmetric|SymmANLS|`symm_anls`|||
||SymmHALS|`symm_halsacc`|||
||SymmNewton|`symm_newton`|||
|Online|INMF|`inmf`|||
||ONMF|`onmf`|||
||Acceralated ONMF|`omf_acc`|||
||SPG|`spg_nmf`|||
||RONMF|`ronmf`|||
||SAGA-MU-NMF|`asag_mu_nmf`|||
||SMU|`smu_nmf`|||
||SVRMU|`svrmu_nmf`|||
|Probabilistic|PNMF-VB|`pnmf_vb`|||
||PNMF-GIBBS|`pnmf_gibbs`|||


<br />

Folders and files
---------
<pre>
./                      - Top directory.
./README.md             - This readme file.
./run_me_first.m        - The scipt that you need to run first.
./demo.m                - Demonstration script to check and understand this package easily. 
./demo_face.m           - Demonstration script to check and understand this package easily. 
|plotter/               - Contains plotting tools to show convergence results and various plots.
|auxiliary/             - Some auxiliary tools for this project.
|solver/                - Contains various optimization algorithms.
    |--- base/          - Basic NMF solvers.
    |--- online/        - Online/stochstic NMF solvers.
    |--- sparse/        - Sparse NMF solvers.
    |--- robust/        - Robust NMF solvers.
    |--- orthogonal/    - Orthogonal NMF solvers.
    |--- symm/          - Symmetric NMF solvers.
    |--- nenmf/         - Nesterov's accelerated NMF solver.
    |--- probabilistic/ - Probabilistic NMF solvers.
    |--- 3rd_party/     - Solvers provided by 3rd_party.
</pre>

<br />                              

First to do
----------------------------
Run `run_me_first` for path configurations. 
```Matlab
%% First run the setup script
run_me_first; 
```

<br />

Simplest usage example: 4 steps!
----------------------------

Just execute `demo` for the simplest demonstration of this package. .

```Matlab
%% Execute the demonstration script
demo; 
```

The "**demo.m**" file contains below.
```Matlab
%% generate synthetic data non-negative matrix V size of (mxn)
m = 500;
n = 100;
V = rand(m,n);
    
%% Initialize rank to be factorized
rank = 5;

%% perform factroization
% MU
options.alg = 'mu';
[w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);
% Hierarchical ALS
options.alg = 'hals';
[w_nmf_hals, infos_nmf_hals] = nmf_als(V, rank, options);        
    
%% plot
display_graph('epoch','cost', {'MU', 'HALS'}, {w_nmf_mu, w_nmf_hals}, {infos_nmf_mu, infos_nmf_hals});

```

<br />

Let's take a closer look at the code above bit by bit. The procedure has only **4 steps**!
<br />

**Step 1: Generate data**

First, we generate  synthetic data of V of size (mxn).
```Matlab    
m = 500;
n = 100;
V = rand(m,n);
```

**Step 2: Define rank**

We set the rank value.
```Matlab
rank = 5;
```

**Step 3: Perform solver**

Now, you can perform optimization solvers, e.g., MU and Hierarchical ALS (HALS), calling [solver functions](#supp_solver), i.e., `nmf_mu()` function and `nmf_als()` function after setting some optimization options. 
```Matlab
% MU
options.alg = 'mu';
[w_nmf_mu, infos_nmf_mu] = nmf_mu(V, rank, options);
% Hierarchical ALS
options.alg = 'hals';
[w_nmf_hals, infos_nmf_hals] = nmf_als(V, rank, options); 
```
They return the final solutions of `w` and the statistics information that include the histories of epoch numbers, cost values, norms of gradient, the number of gradient evaluations and so on.

**Step 4: Show result**

Finally, `display_graph()` provides output results of decreasing behavior of the cost values in terms of the number of iterrations (epochs) and time [sec]. 
```Matlab
display_graph('epoch','cost', {'MU', 'HALS'}, {w_nmf_mu, w_nmf_hals}, {infos_nmf_mu, infos_nmf_hals});
display_graph('time','cost', {'MU', 'HALS'}, {w_nmf_mu, w_nmf_hals}, {infos_nmf_mu, infos_nmf_hals});
```

That's it!


<img src="http://www.kasailab.com/public/github/NMFLibrary/images/cost.png" width="600">

<br />

More plots
----------------------------

"**demo_face.m**" illustrates the learned basis (dictrionary) in case of [CBCL face datasets](http://cbcl.mit.edu/software-datasets/FaceData2.html).

The dataset is first loaded into V instead of generating synthetic data in **Step 1**.

```Matlab
V = importdata('./data/CBCL_face.mat');
```

Then, we can display basis elements (W: dictionary) obtained with different algorithms additionally in **Step 4**.

```Matlab
plot_dictionnary(w_nmf_mu.W, [], [7 7]); 
plot_dictionnary(w_nmf_hals.W, [], [7 7]); 
```

<img src="http://www.kasailab.com/public/github/NMFLibrary/images/face_dictionary.png" width="600">


<br />

License
-------
- The NMFLibrary is **free**, **non-commercial** and **open** source.
- The code provided iin NMFLibrary should only be used for **academic/research purposes**.
- Third party files are included.
    - For ANLS algorithms: `nnlsm_activeset.m`, `nnls1_asgivens.m`, `nnlsm_blockpivot.m`, and `normalEqComb.m` written by Jingu Kim.
    - For PGD algorithm: `nlssubprob.m`.
    - For GNMF algorithm: `GNMF.m`, `GNMF_Multi.m`, `constructW.m` and `litekmeans.m` writtnen by Deng Cai.
    - For SDNMF algorithm: `SDNMF.m`, and `SDNMF_Multi.m` writtnen by Wei Qian.
    - For symmetric algorithms writtnen by D.Kang et al. and Z. Zhu et al.
    - For acceleration sub-routines in [`nmf_mu.m`](https://github.com/hiroyuki-kasai/NMFLibrary/blob/master/solver/base/nmf_mu.m) and [`nmf_als.m`](https://github.com/hiroyuki-kasai/NMFLibrary/blob/master/solver/base/nmf_als.m) for MU and HALS from [Nicolas Gillis](https://sites.google.com/site/nicolasgillis/publications).
    - For dictionaly visualization: `plot_dictionnary.m`, `rescale.m`, and `getoptions.m`.


<br />

Acknowledge
---------------------
- Thank you for big contributions to this library to
    - Haonan Huang

<br />


Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://kasai.comm.waseda.ac.jp/kasai/) (email: hiroyuki **dot** kasai **at** waseda **dot** jp)


<br />

Release notes
--------------
* Version 1.8.1 (Oct. 14, 2020) 
    - Bug fixed in nmf_sc.m and semi_nmf, and added the LPinitSemiNMF algorithm into generate_init_factors.m (Thanks to Haonan Huang). 
* Version 1.7.0 (June 27, 2019) 
    - Symmetic solvers are added.
    - Clustering quality measurements are integrated into store_nmf_infos.m. 
* Version 1.7.0 (May 21, 2019) 
    - PNMF-VB and NeNMF are added.
    - Fixed some bugs. 
* Version 1.6.0 (May 16, 2019) 
    - DTPP is added.
* Version 1.5.1 (Apr. 22, 2019) 
    - Some solvers are modified to fix bugs.
* Version 1.5.0 (Jul. 30, 2018)
    - fnsNMF and NMF-HALS-SO are added.
* Version 1.4.0 (Jul. 24, 2018)
    - sparseMU and orthMU are added.
    - MU with Kullback-Leibler divergence (KL), Amari alpha divergence, and beta divergenceare added.
* Version 1.3.0 (Jul. 23, 2018)
    - NMFsc, scNMF and csNMF are added.
* Version 1.2.0 (Jul. 21, 2018)
    - GNMF, Semi-NMF and SDNMF are added.
* Version 1.1.0 (Apr. 17, 2018)
    - Online/stochastic solvers are added.
* Version 1.0.0 (Apr. 04, 2017)
    - Initial version.

