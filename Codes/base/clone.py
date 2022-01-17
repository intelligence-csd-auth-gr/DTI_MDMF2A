import copy
import warnings

def clone(model, safe=True):
    """ 
    Constructs a new DTI Model with the same parameters.
    Clone does a deep copy of the model without actually copying attached data. 
    It yields a new model with the same parameters that has not been fit on any data.
    
    More detail please see this link:
    Please https://github.com/scikit-learn/scikit-learn/blob/11934e1838b87936da98de31350d88158a8640bb/sklearn/base.py#L39
    
    Parameters
    ----------
    model : DTIModelBase object, or list, tuple or set of objects
        The model or group of models to be cloned
    safe : bool, default=True
        If safe is false, clone will fall back to a deep copy on objects
        that are not models.
    """
    model_type = type(model)
    if model_type in (list, tuple, set, frozenset):
        return model_type([clone(e, safe=safe) for e in model])
    elif not hasattr(model, 'get_params') or isinstance(model, type):
        if not safe:
            return copy.deepcopy(model)
        else:
            if isinstance(model, type):
                raise TypeError("Cannot clone object. " +
                                "You should provide an instance of " +
                                "DTIModel instead of a class.")
            else:
                raise TypeError("Cannot clone object '%s' (type %s): "
                                "it does not seem to be a scikit-learn "
                                "DTIModel as it does not implement a "
                                "'get_params' method."%(repr(model), type(model)))

    klass = model.__class__
    new_object_params = model.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'either does not set or modifies parameter %s'%
                               (model, name))
    return new_object

