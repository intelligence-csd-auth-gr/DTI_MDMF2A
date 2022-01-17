
class MultiViewInductiveModelBase():
    """Base class providing API and common functions for all multi-view Inductive DTI prediction model.

    Attributes
    ----------
    copyable_attrs : List[str]
        list of attribute names that should be copied when class is cloned
    """

    def __init__(self):
        self.copyable_attrs = []


    def _check_fit_data(self, intMat, drugMats, targetMats, cvs):
        """Check the input dataset for training. Raises RuntimeError if it is not right.
        """
        self._n_dsims = drugMats.shape[0]
        self._n_tsims = targetMats.shape[0]
        # if dsim_names.size != self._n_dsims:
        #     raise RuntimeError("The dsim_names (size: {}) does not fit for drugMats (shape: {})!!".format(dsim_names.size, drugMat.shape))
        # if tsim_names.size != self._n_tsims:
        #     raise RuntimeError("The tsim_names (size: {}) does not fit for targetMats (shape: {})!!".format(dsim_names.size, drugMat.shape))
        
        if drugMats[0].shape[0] != drugMats[0].shape[1]:
            raise RuntimeError("The drugMat (shape: {}) is not square matrix!!".format(drugMats[0].shape))
        if targetMats[0].shape[0] != targetMats[0].shape[1]:
            raise RuntimeError("The targetMat (shape: {}) is not square matrix!!".format(targetMats[0].shape))
        self._n_drugs, self._n_targets = intMat.shape
        if self._n_drugs != drugMats[0].shape[0]:
            raise RuntimeError("The drugMat (shape: {}) is not comparable with intMat's row {}".format(drugMats[0].shape,intMat.shape[0]))
        if self._n_targets != targetMats[0].shape[0]:
            raise RuntimeError("The targetMat (shape: {}) is not comparable with intMat's column {}".format(targetMats[0].shape,intMat.shape[1]))
        if cvs == 2 or cvs == 3 or cvs == 4:
            self._cvs = cvs
        else:
            raise RuntimeError("The cvs = {} is illegal, it should be 2, 3 or 4!!".format(cvs))
    
    
    def _check_test_data(self, drugMatTes, targetMatTes):
        """Check if the input dataset for predicton is right. Raises RuntimeError if it is not right.
        """
        if drugMatTes.shape[0] != self._n_dsims:
            raise RuntimeError("The number of similairties in drugMatTes ({}) is not same with self._n_dsims ({})!!".format(drugMatTes.shape[0], self._n_dsims))
        if targetMatTes.shape[0] != self._n_tsims:
            raise RuntimeError("The number of similairties in targetMatTes ({}) is not same with self._n_dsims ({})!!".format(targetMatTes.shape[0], self._n_tsims))
        
        if self._n_drugs != drugMatTes[0].shape[1]:
            raise RuntimeError("The n_drugs:{} is not comparable with drugMatTe's column {}!!".format(self._n_drugs,drugMatTes.shape[1]))
        if self._n_targets != targetMatTes[0].shape[1]:
            raise RuntimeError("The n_targets:{} is not comparable with targetMatTe's column {}!!".format(self._n_targets,targetMatTes.shape[1]))
        self._n_drugs_te = drugMatTes[0].shape[0]
        self._n_targets_te = targetMatTes[0].shape[0]
        if self._cvs == 2 and self._n_targets != self._n_targets_te:
            raise RuntimeError("For cvs=2, _n_targets({}) != _n_targets_te({}) !!".format(self._n_targets,self._n_targets_te))
        if self._cvs == 3 and self._n_drugs != self._n_drugs_te:
            raise RuntimeError("For cvs=2, _n_drugs({}) != _n_drugs_te({}) !!".format(self._n_drugs,self._n_drugs_te))
 
            

        
        

    def fit(self, drugMats, TargetMats, intMat, csv=2):
        """Abstract method to fit model with training dataset
        It must return a fitted instance of :code:`self`.

        Parameters
        ----------
        drugMats : numpy.ndarray shape (n_dsims, n_drugs, n_drugs)
        TargetMat : numpy.ndaarray shape (n_tsims, n_targets, n_targets)
        intMat: numpy.ndaarray shape (n_drugs, n_targets)
        cvs: the CV setting, the value of cvs is from {2,3,4}
        
        Returns
        -------
        object: fitted instance of self

        Raises
        ------
        NotImplementedError
            this is just an abstract method
        """

        raise NotImplementedError("DTIModelBase::fit()")

    def predict(self, drugMatTes, targetMatTes):
        """Abstract method to predict labels

        Parameters
        ----------
        drugMatTe: numpy.ndarray shape (n_dsims, n_drugs_te, n_drugs)
        TargetMatTe: numpy.ndarray shape (n_dsims, n_targets, n_targets_te)

        Returns
        -------
        predicting socres: numpy.ndarray shape (n_drugs_te, n_drugs)

        Raises
        ------
        NotImplementedError
            this is just an abstract method
        """
        raise NotImplementedError("DTIModelBase::predict()")

    def get_params(self, deep=True):
        """Get parameters to sub-objects

        Introspection of classifier for search models like
        cross-validation and grid search.

        Parameters
        ----------
        deep : bool
            if :code:`True` all params will be introspected also and
            appended to the output dictionary.

        Returns
        -------
        out : dict
            dictionary of all parameters and their values. If 
            :code:`deep=True` the dictionary also holds the parameters
            of the parameters.
        """
        out = dict()

        for attr in self.copyable_attrs:
            out[attr] = getattr(self, attr)

            if hasattr(getattr(self, attr), 'get_params') and deep:
                deep_items = list(getattr(self, attr).get_params().items())
                out.update((attr + '__' + k, val) for k, val in deep_items)

        return out

    def set_params(self, **parameters):
        """Propagate parameters to sub-objects

        Set parameters as returned by :code:`get_params`. Please 
        see this `link`_.

        .. _link: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py#L243
        """

        if not parameters:
            return self

        valid_params = self.get_params(deep=True)

        parameters_current_level = [x for x in parameters if '__' not in x]
        for parameter in parameters_current_level:
            value = parameters[parameter]

            if parameter in valid_params:
                setattr(self, parameter, value)
            else:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (parameter, self))

        parameters_below_current_level = [x for x in parameters if '__' in x]
        parameters_grouped_by_current_level = {object: {} for object in valid_params}

        for parameter in parameters_below_current_level:
            object_name, sub_param = parameter.split('__', 1)

            if object_name not in parameters_grouped_by_current_level:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (object_name, self))

            value = parameters[parameter]
            parameters_grouped_by_current_level[object_name][sub_param] = value

        valid_params = self.get_params(deep=True)

        # parameters_grouped_by_current_level groups valid parameters for subojects
        for object_name, sub_params in parameters_grouped_by_current_level.items():
            if len(sub_params) > 0:
                sub_object = valid_params[object_name]
                sub_object.set_params(**sub_params)

        return self
