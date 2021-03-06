 - GraphonEst:
 -------------

	Python package to accomplish the graphon estimation routine from 'EM-based smooth 
	Graphon Estimation using Bayesian and Spline based Approaches' [arXiv:1903.06936].

	* dependencies [(~ ...) = included in python standard library]:
		(~ sys)
		(~ os)
		(~ copy)
		(~ warnings)
		(~ operator)
		~ matplotlib
		~ numpy
		~ scipy
		~ math
		(~ csv)
		(~ pickle)
		~ cvxopt
		~ networkx
		~ sklearn
		~ mpl_toolkits


 - application.py:
 -----------------

	-> File to execute for running the algorithm.

	 * command in terminal / shell / console:
	   python3 >path_to_file</application.py

	 * adjustment when running in interactive session:
	   dir1_ = os.path.dirname(os.path.realpath(''))

	 * Creates new local folder named 'Graphics'
	   where all figures will be saved.


 - Data:
 -------

	Data files of different real world network examples.

	* facebook/0.edges

		facebook ego network data set, downloaded from 

			http://snap.stanford.edu/data [online; accessed 06-14-2018].
			[Leskovec, J. and A. Krevl (2014). SNAP Datasets: Stanford Large Network Dataset 
			Collection.]

	* alliances/alliances_strong_post_adjMat_2016.csv

		network data of military alliances (processed), downloaded from

			http://www.atopdata.org/data [online; accessed 03-02-2020].
			[Leeds, B. A., J. M. Ritter, S. McLaughlin Mitchell and A. G. Long (2002). Alliance 
			Treaty Obligations and Provisions, 1815-1944. International Interactions 28: 237-260.]

	* human_brain/Coactivation_matrix.mat

		human brain functional coactivation network, downloaded from

			https://sites.google.com/site/bctnet/datasets [online; accessed 10-15-2019].
			[Rubinov, M. and Sporns, O. (2010). Brain Connectivity Toolbox (MATLAB).]

			and collected from

			[Crossley, N. A., Mechelli, A., Vértes, P. E., Winton-Brown, T. T., Patel, A. X., 
			Ginestet, C. E., McGuire, P. and Bullmore, E. T. (2013). Cognitive relevance of the 
			community structure of the human brain functional coactivation network. Proceedings 
			of the National Academy of Sciences 110 11583–11588.]


