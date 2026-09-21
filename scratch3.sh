for hash in 56e4dc1e983d7917a11faca8780eaac17ad6dfae 986364fc3d4510cee27a2c50509a03de2431db38 a0429d4909717c174dc902e56825a7317c410cd2 eef7d1a9a2fe255f5f92f3805ff1c4695170af8d; do
  echo "Looking for $hash"
  for commit in $(git fsck --lost-found 2>/dev/null | grep commit | awk '{print $3}'); do
    if git ls-tree -r $commit | grep -q $hash; then
      echo "Found in $commit:"
      git ls-tree -r $commit | grep $hash
      break
    fi
  done
done
