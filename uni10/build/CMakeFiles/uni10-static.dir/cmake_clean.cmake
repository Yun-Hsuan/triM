FILE(REMOVE_RECURSE
  "libuni10.pdb"
  "libuni10.a"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/uni10-static.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
