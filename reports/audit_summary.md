# Dataset audit summary

## Counts per class

| class     |   files |   readable |   corrupt |   duplicates |
|:----------|--------:|-----------:|----------:|-------------:|
| AiArtData |     538 |        538 |         0 |            1 |
| RealArt   |     435 |        435 |         0 |            0 |


## Width / height / file size (readable images)

| class     |   ('width', 'min') |   ('width', 'median') |   ('width', 'max') |   ('height', 'min') |   ('height', 'median') |   ('height', 'max') |   ('file_size', 'min') |   ('file_size', 'median') |   ('file_size', 'max') |
|:----------|-------------------:|----------------------:|-------------------:|--------------------:|-----------------------:|--------------------:|-----------------------:|--------------------------:|-----------------------:|
| AiArtData |                225 |                1028.5 |               6000 |                 135 |                    800 |                4001 |                  11738 |                    199802 |            5.62618e+06 |
| RealArt   |                183 |                 940   |               8495 |                 158 |                    720 |                7091 |                  11311 |                    144982 |            1.01954e+07 |


## Formats and colour modes

| class     |   JPEG |   PNG |
|:----------|-------:|------:|
| AiArtData |    428 |   110 |
| RealArt   |    402 |    33 |


| class     |   L |   P |   RGB |   RGBA |
|:----------|----:|----:|------:|-------:|
| AiArtData |   1 |  14 |   488 |     35 |
| RealArt   |   6 |   9 |   406 |     14 |


- Total files: 973
- Corrupt / unreadable: 0
- Exact duplicate files (extra copies): 1
- Duplicate groups spanning BOTH classes (label conflicts): 0
