FILE(REMOVE_RECURSE
  "libuni10.pdb"
  "libuni10.so"
  "libuni10.so.1.0.0"
  "libuni10.so.1"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/uni10.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
