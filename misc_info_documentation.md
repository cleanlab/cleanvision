misc_info: dict
a dictionary where keys are miscellanous info name strings and values are an intuitive data structure containing that info.

The following documentation states the miscellaneous info stored in misc_info associated with each issue check.

"Brightness":
    - 'Brightness sorted z-scores' (list[numpy.ndarray]): a list of two-element numpy arrays where the first element in each array is the image index, the second is the Brightness z-score. Arrays sorted based on z-score from low to high. 

"Odd Size":
    - 'Odd Size sorted z-scores' (list[numpy.ndarray]): a list of two-element numpy arrays where the first element in each array is the image index, the second is the Odd Size z-score. Arrays sorted based on z-score from low to high. 

"Potential Occlusion": 
    - 'Potential Occlusion sorted z-scores' (list[numpy.ndarray]): a list of two-element numpy arrays where the first element in each array is the image index, the second is the Potential Occlusion z-score. Arrays sorted based on z-score from low to high. 

"Potential Static": 
    - 'Potential Static sorted z-scores' (list[numpy.ndarray]): a list of two-element numpy arrays where the first element in each array is the image index, the second is the Potential Static z-score. Arrays sorted based on z-score from low to high. 

"Blurry":
    - 'Blurry sorted z-scores' (list[numpy.ndarray]): a list of two-element numpy arrays where the first element in each array is the image index, the second is the Blurry z-score. Arrays sorted based on z-score from low to high. 

"Duplicated": 
    - 'Image Hashes'(set): A set of md5 hashes of all images in the dataset.

    - 'Hash to Image'(dict): A dictionary where the keys are md5 hashes, and respective values are a list of all image names that yielded to that hash. 

    - 'Duplicate Image Groups'(dict): Contains key-value pairs from 'Hash to Image' where the duplicates issue is present (the list of image names with a particular md5 hash is at least 2). 

"Near Duplicates":
    - 'Near Duplicate Imagehashes': A set of imagehashes (default = phash) of all images in the dataset.

    - 'Imagehash to Image': A dictionary where the keys are imagehashes, and respective values are a list of all image indices that yielded to that hash. 

    - 'Near Duplicate Image Groups': Contains key-value pairs from 'Imagehash to Image' where the near duplicates issue is present (the list of image indices with a particular imagehash is at least 2). 