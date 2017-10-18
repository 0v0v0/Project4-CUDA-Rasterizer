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

![result](pic/toon_actual.png)

![result](pic/toon_debug.png)
    
### 2. Sketch Shader

![result](pic/sketch_a_duck.png)

In fact, this shader is inspired by one of the many bugs I have. 

![result](pic/worng_texture.png)

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

