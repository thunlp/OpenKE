# conceptnet2openke
An end-2-end python3 script to convert conceptnet assertions into training file format of OpenKE

### Usage:

```shell
python3 conceptnet2openke.py cache_folder_path language topn_ents output_folder
        
@params:
cache_folder_path: folder path to download and store the conceptnet assertions file
language: the language type of the concepts to extract (eg. en, ru, uk, etc)
topn_ents: number of top-most popular concepts to consider
output_folder: folder path to store OpenKE consumable files, i.e., entity2id.txt, 
               relation2id.txt, train/valid/test2id.txt
        
```

