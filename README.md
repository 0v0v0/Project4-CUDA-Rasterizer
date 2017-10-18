CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* JuYang
* ### Tested on: Windows 7, i7-4710MQ @ 2.50GHz 8GB, GTX 870M 6870MB (Hasee Notebook K770E-i7)

![result](pic/sketch_fin.gif)

## HighLight Features

### 1. Toon Shader
![result](pic/toon_actual.gif)

![result](pic/toon_debug.gif)
    
I made a toon shader with outline drawing.

In my toon shader function, I requested one more parameter: float layers. 

This parameter is used to define how many segments the toon shader will have. 

When passing to the toon shader, the color space is continuous from 0 to 1, then it will be divided into several segments. 

For example, if we have 5 segments, then colors from 0 to 0.2 falls to the first segment, and colors from 0.2 to 0.4 falls to the second segment. 

This step is simple, next is to draw the lines. 

If you look at "borderland", which is a successful toon shader game, you will find out that there're 2 kinds of lines: 

the lines that surrounds the object, called "outer line". 

and the lines that appears within the object(usually used to illustrate important geomitry, like nose and eyes and lips). I call them "inner line"

The technique to draw outer lines is simple. Insdead of checking dot(pixel.normal,camera.foward) changes from positive to negative, we can simply check the depth buffer. If the depth differs a lot, larger than the threshold, we can assume that this is a boundary. 

And about inner lines. Since inner lines present the sharp edge of geomitry, or the boundaries between 2 objects(eg. eyeballs and eyelids)， we should check the normal differents. If the normal differs a lot at a point, larger than the normal's threshold, we can say that this is a inner line boundary.

![result](pic/toon_debug.png)

This picture presents the debugging mode for outlines drawing. The red line is the outer line, and the green line is the inner line. You can see that inner lines appears as assisting lines that helps to draw the geomtries. 

![result](pic/toon_actual.png)
    
### 2. Sketch Shader

![result](pic/sketch_a_duck.png)

In fact, this shader is inspired by one of the many bugs I have. 

![result](pic/wrong_texture.png)

Then, it reminds me of a kind of comic books that's once popular in China during the 1960s

![result](pic/xiaorenshu.png)

When I was a child, I have a lot of these comic books that my father read when he was a child. Well, now I know this is called pen drawing, a complicated art. 

Then I tried to mock this art form, and this is my first trial.

![result](pic/basic_sketch.png)

In this implimentation, I only drawed the color gradiant outline from toon shader. But when looking at this, I think something is wrong. 
Since I have no idea about drawing, I asked a friend who's an art student. 

The law of pen drawing is: 

1- always draw lines in same width. 

2- using line density to present surface color intensity. (The darker an area is, the more lines it will have)

3- lines should follow the surface's geomtry. 

![result](pic/pen_draw.png)

I think this could illustrate that well. 
