	README
	
	HOW TO RUN main.py

	Need to put the python file in the same folder as the source picture.
	To run this program, there are four argument. 
	The first one is the path of the image.
	Second one can be 'raw' or 'edge'. If using 'raw' then the program will use raw RGB alignment, 
while using 'edge' will make the program run with edge detection alignment, which has a much better result.
	Third one can be 'yes' or 'no', indicate whether or not to use Histogram Equalization to
improve the quality of the image. 
	Fourth one can also be 'yes' or 'no', indicate whether ot not to use auto cropping at the end 
of the procedure.

	For example, the input "python main.py cathedral.jpg edge yes yes" would produce a colorized 
image from cathedral.jpg and store it in the current directory. The image produced is aligned with edge detection technique and improved by Histogram Equalization algorithm, and cropped automatically before being stored. 

