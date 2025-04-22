from typing import Any

class Wrapper:
    """
    Wrapper class for wrapping objects and inheriting their properties
    """
    def __init__(self, base_object: Any):
        """Claims the base object type and adds its properties to the current class

        :param base_object: Object to wrap
        :type base_object: Any
        """        
        new_cls_dict = {}
        for k, v in type(base_object).__dict__.items():
            if type(v) == property:
                new_cls_dict[k] = v
        self.__class__ = type(self.__class__.__name__, (self.__class__, ), {
            **type(self).__dict__,
            **new_cls_dict
        })
        self._base_object = base_object

    def __getattr__(self, name: str) -> Any:
        """Gets the class attributes

        :param name: The name of attribute to be gotten
        :type name: Any
        :return: The name of the current class if it exists, otherwise the name of the wrapped obj
        :rtype: Any
        """
        if name in self.__dict__:
            return getattr(self, name)
        return getattr(self._base_object, name)