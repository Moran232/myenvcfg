# myenvcfg
vimrc and tmux config files for Linux
use Vundle to install plugins

update vim to 8.2 or higher first
```
  apt-get install software-properties-common (when add-apt-repository failed)
  sudo add-apt-repository ppa:jonathonf/vim
  sudo apt update
  sudo apt install vim
```

0  clone this repo and mv .vimrc to ~/

1  clone Vundle to manage plugin
```
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```
2 open vim and exec in CLI
```
:BundleInstall
```

3  mv colors to .vim/ as molokai cant be installed.

4  (optional) install tmux: suao apt-get install tmux and mv .tmux.conf to ~/

5  (optional) install termdebug
```
:packadd termdebug
```
