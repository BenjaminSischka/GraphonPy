 - GraphonEst:
 -------------

	Python package to conduct the graphon estimation routine from 'EM-Based 
	Smooth Graphon Estimation Using MCMC and Spline-Based Approaches'. 
	[https://doi.org/10.1016/j.socnet.2021.08.007]

	* dependencies [(~ ...) = included in python standard library]:
		(~ sys)
		(~ os)
		(~ copy)
		(~ warnings)
		(~ operator)
		(~ csv)
		(~ pickle)
		~ matplotlib
		~ numpy
		~ scipy
		~ math
		~ cvxopt
		~ networkx
		~ sklearn
		~ mpl_toolkits


 - application.py:
 -----------------

	-> File to execute for running the algorithm.

	 * command in terminal / shell / console:
	   python3 >>path_to_file<</application.py > >>path_to_file<</output.txt

	 * adjustment when running in interactive session:
	   dir1_ = os.path.dirname(os.path.realpath(''))

	 * Creates new local folder named 'Graphics' where all figures will be saved.


 - Data:
 -------

	Data sources which have been used for real world network examples.


	* facebook ego network data set, downloaded from and assembled by

			http://snap.stanford.edu/data [online; accessed 06-14-2018].
			[Leskovec, J. and A. Krevl (2014). SNAP Datasets: Stanford Large Network Dataset 
			Collection.]


	* military alliance network (processed), downloaded from and assembled by

			http://www.atopdata.org/data [online; accessed 03-02-2020].
			[Leeds, B. A., J. M. Ritter, S. McLaughlin Mitchell and A. G. Long (2002). Alliance 
			Treaty Obligations and Provisions, 1815-1944. International Interactions 28: 237-260.]


	* human brain functional coactivation network, downloaded from

			https://sites.google.com/site/bctnet/datasets [online; accessed 10-15-2019].
			[Rubinov, M. and Sporns, O. (2010). Brain Connectivity Toolbox (MATLAB).]

			and assembled by

			[Crossley, N. A., Mechelli, A., Vértes, P. E., Winton-Brown, T. T., Patel, A. X., 
			Ginestet, C. E., McGuire, P. and Bullmore, E. T. (2013). Cognitive relevance of the 
			community structure of the human brain functional coactivation network. Proceedings 
			of the National Academy of Sciences 110 11583–11588.]
