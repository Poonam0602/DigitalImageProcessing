# DigitalImageProcessing
Measuring Objectness of Image windows -> Project Demo

*Project Demonstration Topic :* Comparison of results for object detection with and without objectness measure. 

*Resullt : This implementation demonstrates that objectness measure of image window reduces the computation which is proven by getting number of windows to generate same results as of without "objectness measure".*

Images | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 
----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
Number of Win w/o objectness measure | 1699 | 1414 | 1414 | 2103 | 1414 | 1414 | 120 | 128 | 128 | 128
Number of Win objectness measure | 31 | 64 | 47 | 51 | 114 | 10 |1 | 1 | 3 | 3

*Folder Structure:*
'''bash
Code-> Bin
    -> data->config
           ->datasets-> CelebA
                     -> test_images
                     -> |WIDER
           ->features-> neg
                     -> pos
           ->models
    -> object_detector
'''    
*Platform Used:*
Linux
OpenCV

*Run Program:*
1.Download the dataset from given linnks and extract those in respective folders.
2. Move to bin folder
3. Run python test_object_detector.py (To get results without measuring objectness of image windows)
4. Run python test_objectness.py (To get results WITH measuring objectness of image windows)



