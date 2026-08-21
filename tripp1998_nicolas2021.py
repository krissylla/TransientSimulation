# for reference

def tripp1998( x1, c,
                   mabs=-19.3, sigmaint=0.10,
                   alpha=-0.14, beta=3.15, rng=None):
        """Get the 2-parameter absolute (natural) SNe Ia magnitude.

        Parameters
        ----------
        x1 : array
            The lightcurve stretch. `x1` and `c` must have the same size.

        c : array
            The lightcurve color. `x1` and `c` must have the same size.

        mabs : float, optional
            The average absolute magnitude at `c=0` and `x1=0`. The default is
            -19.3.

        sigmaint : float, optional
            The scale of the normal grey scatter (on `mabs`). The default is
            0.10.

        alpha : float, optional
            The stretch linear law coefficient. The default is -0.14.

        beta : float, optional
            The color linear law coeeficient. The default is 3.15.

        rng: None, int, Generator
            Random number generator seed. 
            (docstring extracted from `np.random.default_rng()`, see this for complete documentation).
            If None, then fresh, unpredictable entropy will be pulled from the OS. 
            If an ``int``, then the seed will start from this.
            If passed a `Generator`, it will be returned unaltered.

        Returns
        -------
        array
           The absolute magnitude, with the same format as `x1` and `c`.
        """
        rng = np.random.default_rng(rng)
        mabs = rng.normal(loc=mabs, scale=sigmaint, size=len(x1))
        mabs_notstandard = mabs + (x1*alpha + c*beta)
        return mabs_notstandard

def nicolas2021( xx="-4:4:0.005", 
                     mu1=0.33, sigma1=0.64, 
                     mu2=-1.50, sigma2=0.58, a=0.45,
                     fprompt=0.5):
        """Get the pdf of the Nicolas (2021) model.

        Parameters
        ----------
        xx : str or array, optional
            Definition range for the parameters. Draws will be done from this
            array given the pdf that will be estimated for it. If a string is
            given, it is evaluated as `np.r_[xx]`. The default is
            "-4:4:0.005".

        mu1 : float, optional
            The mean of the first gaussian. The default is 0.33.

        sigma1 : float, optional
            The standard deviation of the first gaussian. The default is 0.64.

        mu2 : float, optional
            The mean of the second gaussian. The default is -1.50.

        sigma2 : float, optional
            The standard deviation of the second gaussian. The default is 0.58.

        a : float, optional
            The relative influence of both modes (1 or 2) in the delayed
            environment. `a>0.5` means more mode 1. The default is 0.45.

        fprompt : float, optional
            The fraction of prompt SNe Ia. This is ignored if `redshift` is
            given. The default is 0.5.

        Returns
        -------
        tuple
            A tuple containing the x-axis and the pdf.
        """
        from scipy.stats import norm
        if type(xx) is str: # assumed r_ input
            xx = eval(f"np.r_[{xx}]")
            
        mode1 = norm.pdf(xx, loc=mu1, scale=sigma1)
        mode2 = norm.pdf(xx, loc=mu2, scale=sigma2)
        if type(fprompt) is not float: 
            fprompt = np.asarray(fprompt)[:,None]
            
        pdf = fprompt*mode1 + (1-fprompt)*(a*mode1 + (1-a)*mode2)
        return xx, pdf