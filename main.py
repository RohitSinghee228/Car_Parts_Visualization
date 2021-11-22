import cv2
import pandas as pd
import numpy as np
import utlis
import random

#Master Function
def master(image_path, meta_data_json_path, opacity=50):
    col_list = []
    list, choice = utlis.meta_data(meta_data_json_path)
    img = cv2.imread(image_path)
    img2 = img.copy()
    base = img.copy()


    img_poly,col_list=utlis.draw_fill_polygons(img2,list,shade=60)

    alpha = opacity / 100
    #add weight function alpha paramenter to set transparency
    cv2.addWeighted(img_poly, alpha, base, 1 - alpha, 0, base)

    #Second Iteration for Bounding boxes
    #Different Loops used so That the bounding rectangle boxes are on top of the the
    # polygons and are not cut off by polygon shapes
    img_bound=utlis.draw_bounding_box(base,list,col_list,shade=60)

    #for Debugging and display Puproses
    # cv2.imshow("Polygon Visual",base)
    # cv2.imshow("Bounding Box",img_bound)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return(base,img_bound)

#loop for iterating over all the images
for k in range(1,6):

    image_path='Data Visualization/images/'+str(k)+'.jpg'
    meta_data_json_path='Data Visualization/data/'+str(k)+'.json'
    img_poly,img_bound=master(image_path, meta_data_json_path)
    cv2.imwrite('Data Visualization/Answer/img_poly'+str(k)+'.jpg',img_poly)
    cv2.imwrite('Data Visualization/Answer/img_bound' + str(k) + '.jpg', img_bound)


