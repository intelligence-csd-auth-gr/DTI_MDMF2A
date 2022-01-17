class Combine_Sims_Base():

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
    #---------------------------------------------------------------------------------------------------

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