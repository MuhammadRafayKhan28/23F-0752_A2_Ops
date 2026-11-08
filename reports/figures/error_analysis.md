# Error analysis - most confident mistakes on the test set

| # | file | true | predicted P(AI) | size | sharpness | saturation | hypothesis |
|---|---|---|---|---|---|---|---|
| 1 | most-famous-paintings-by-jan-van-eyck.jpg | Real | 0.98 | 1200x690 | 2094 | 51 | heavy high-frequency texture/noise (detailed or over-sharpened) |
| 2 | picturesque-scenery-waterfall-flowing-through-260nw-2360180703.jpg | Real | 0.92 | 390x280 | 691 | 51 | small source image, upscaling to 224 adds blur |
| 3 | Indigenous-Peoples-Myanmar-Meriem-Gray--World-Bank-1440x600.png | Real | 0.90 | 1440x600 | 2693 | 65 | heavy high-frequency texture/noise (detailed or over-sharpened); extreme aspect ratio distorted by stretching to a square |
| 4 | Bu69yrI45ueNWQX84ap5.jpg | AI | 0.17 | 1024x768 | 815 | 62 | photorealistic generator output with natural lighting |
| 5 | sunset-trees-abstract-painting-3.jpg | Real | 0.82 | 800x800 | 937 | 65 | painting/illustration style or studio-like lighting resembles the other class |
| 6 | tiger4.jpg | AI | 0.19 | 3733x2133 | 1851 | 70 | heavy high-frequency texture/noise (detailed or over-sharpened) |
| 7 | Midjourney-1-9527a04.jpg | AI | 0.20 | 811x645 | 1503 | 20 | heavy high-frequency texture/noise (detailed or over-sharpened) |
| 8 | nature-scenery.jpg | Real | 0.78 | 698x699 | 784 | 61 | painting/illustration style or studio-like lighting resembles the other class |
| 9 | famous-portrait-paintings_wide-3-2-636b81a7177ce.png | Real | 0.77 | 1200x800 | 1025 | 48 | painting/illustration style or studio-like lighting resembles the other class |
| 10 | AI_art_1685017093404_1685017098933.jpeg | AI | 0.24 | 1140x641 | 1579 | 36 | heavy high-frequency texture/noise (detailed or over-sharpened) |
