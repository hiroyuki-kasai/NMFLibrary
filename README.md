# NMFLibrary
Matlab library for non-negative matrix factorization (NMF)

Authors: [Hiroyuki Kasai](http://www.kasailab.com/)

Last page update: April 04, 2017

Latest library version: 1.0.0 (see Release notes for more info)

<br />

Introduction
----------
The NMFLibrary is a **pure-Matlab** library of a collection of algorithms of **non-negative matrix factorization (NMF)**. 

<br />

## <a name="supp_solver"> List of the algorithms available in NMFLibrary </a>


- **MU** (multiplicative updates)
    - MU
        - Daniel D. Lee and H. Sebastian Seung, "[Algorithms for non-negative matrix factorization](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf)," NIPS 2000.
    - Modified MU
    - Acceralated MU

- **PGD** (projected gradient descent)
    - PGD
    - Direct PGD

- **ALS** (alternative least squares)
    - ALS
    - Hierarchical ALS

- **ANAS** (alternative non-negative least squares)
    - ASGROUP (ANLS with Active Set Method and Column Grouping)
    - ASGIVENS (ANLS with Active Set Method and Givens Updating)
    - BPP (ANLS with Block Principal Pivoting Method)


<br />

## Algorithm configurations


|Algorithm name in example codes| function | `options.alg` | other `options` |
|---|---|---|---|
|MU|`nmf_mu`|`mu`||
|Acceralated MU|`nmf_mu`|`acc_mu`||
|ALS|`nmf_als`|`als`||
|Hierarchical ALS|`nmf_als`|`hals_mu`||
|Acceralated hierarchical ALS|`nmf_als`|`acc_hals_mu`||
|PGD|`nmf_pgd`|`pgd`||
|Direct PGD|`nmf_pgd`|`direct_pgd`||
|ASGROUP|`nmf_anls`|`anls_asgroup`||
|ASGIVENS|`nmf_anls`|`anls_asgivens`||
|BPP|`nmf_anls`|`anls_bpp`||


<br />

Folders and files
---------

- run_me_first.m
    - The scipt that you need to run first.

- demo.m and demo_face.m
    - Demonstration scripts to check and understand this package easily. 
                    
- plotter/
    - Contains plotting tools to show convergence results and various plots.
                  
- auxiliary/
    - Some auxiliary tools for this project.

- solver/
    - Contains various optimization algorithms.

                  
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
 %% generate synthetic data of (mxn) matrix       
m = 500;
n = 100;
V = rand(m,n);
    
%% Initialize of rank to be factorized
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
Let take a closer look at the code above bit by bit. The procedure has only **4 steps**!

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

Now, you can perform optimization solvers, i.e., MU and Hierarchical ALS (HALS), calling [solver functions](#supp_solver), i.e., `nmf_mu()` function and `nmf_als()` function after setting some optimization options. 
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


<img src="https://dl.dropboxusercontent.com/u/869853/github/NMFLibrary/images/cost.png" width="600">

<br />

More plots
----------------------------

"**demo_face.m**" illustrates the learned basis (dictrionary) in case of [CBCL face datasets](http://cbcl.mit.edu/software-datasets/FaceData2.html).

```Matlab
%% display basis elements obtained with different algorithms
plot_dictionnary(w_nmf_mu.W, [], [7 7]); 
plot_dictionnary(w_nmf_hals.W, [], [7 7]); 
```

<img src="https://dl.dropboxusercontent.com/u/869853/github/NMFLibrary/images/face_dictionary.png" width="600">


<br />

License
-------
- The NMFLibrary is **free**, **non-commercial** and **open** source.
- The code provided iin NMFLibrary should only be used for **academic/research purposes**.
- Third party files are included.
    - For ANLS algorithms `nnlsm_activeset.m`, `nnls1_asgivens.m`, `nnlsm_blockpivot.m`, and `normalEqComb.m`.
    - For PGD algorithm: `nlssubprob.m`.
    - For dictionaly illustrations: `plot_dictionnary.m`, `rescale.m`, and `getoptions.m`.


<br />

Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://www.kasailab.com/) (email: kasai **at** is **dot** uec **dot** ac **dot** jp)

<br />

Release Notes
--------------

* Version 1.0.0 (Apr. 04, 2017)
    - Initial version.

