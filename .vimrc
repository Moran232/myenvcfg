"==================================

"    Vim基本配置

"===================================

"==============显示设置start=============
""关闭vi的一致性模式 避免以前版本的一些Bug和局限
set nocompatible

"配置backspace键工作方式
set backspace=indent,eol,start

"显示行号
set number

"设置在编辑过程中右下角显示光标的行列信息
set ruler

"设置搜索高亮
set hlsearch

"设置命令行的文本补全
set wildmenu

"设置窗口缩放
set modifiable

"在状态栏显示正在输入的命令
set showcmd

"突出现实当前行列
set cursorline
"set cursorcolumn

"设置C/C++方式自动对齐
set autoindent
set cindent

"开启语法高亮功能
syntax enable

"设置搜索时忽略大小写
set ignorecase

"==============显示设置over==================

"==============编辑设置start=================

"设置历史记录条数
set history=1000

"设置取消备份 禁止临时文件生成
set nobackup
set noswapfile

"设置匹配模式 类似当输入一个左括号时会匹配相应的那个右括号
set showmatch

"设置在Vim中可以使用鼠标 防止在Linux终端下无法拷贝
set mouse=a

"设置Tab宽度
set tabstop=4
"设置自动对齐空格数
set shiftwidth=4
"设置按退格键时可以一次删除4个空格
"set smarttab
"将Tab键自动转换成空格 真正需要Tab键时使用[Ctrl + V + Tab]
set expandtab
"设置编码方式
set encoding=utf-8
"自动判断编码时 依次尝试一下编码
set fileencodings=ucs-bom,utf-8,cp936,gb18030,big5,euc-jp,euc-kr,latin1
"针对不同的文件采用不同的缩进方式
filetype indent on

"=============编辑设置over===================


"==================================
"       leader 键设置
"==================================
let mapleader = "\<Space>"
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>
nnoremap <leader>p :bp<CR>
nnoremap <leader>n :bn<CR>
"==================================
"termdebug 设置
"==================================
packadd termdebug
let g:termdebug_wide = 163

"=========================================
"    Vundle及其插件设置
"    安装：:BundleInstall
"    卸载：注释插件后:BundleClean
"==========================================

"允许插件
filetype plugin on
"检测文件类型
filetype off "requied
"Vundle配置必须 开启插件
filetype plugin indent on

"=================插件安装=================
set rtp+=~/.vim/bundle/Vundle.vim
"call vundle#rc()
call vundle#begin()
"使用Vundle来管理Vundle
Plugin 'gmarik/vundle'

"PowerLine插件 状态栏增强展示
Plugin 'Lokaltog/vim-powerline'

"安装NERD-tree
Plugin 'preservim/nerdtree'
"nerdtree 文件中git支持
Plugin 'Xuyuanp/nerdtree-git-plugin'
"多个窗口共享一个nerdtree
Plugin 'jistr/vim-nerdtree-tabs'
"快速注释  leader c+scape  leader cc leader cu
Plugin 'scrooloose/nerdcommenter'

"显示vertical线
Plugin 'Yggdroot/indentLine'

"代码补全
Plugin 'Valloric/YouCompleteMe'

Plugin 'jiangmiao/auto-pairs'

Plugin 'Chiel92/vim-autoformat'
nnoremap <F6> :Autoformat<CR>
let g:autoformat_autoindent = 0
let g:autoformat_retab = 0
let g:autoformat_remove_trailing_spaces = 0

call vundle#end()            " required
filetype plugin indent on    " required
"===============monokai主题配色=====================
set t_Co=256
let g:molokai_original = 1 "prefer the scheme to match the original monokai background
let g:rehash256 = 1
set term=screen-256color
colorscheme molokai
"===============ycm设置==========================
" 禁止YCM 自动弹出函数原型预览窗
set completeopt-=preview
let g:ycm_add_preview_to_completeopt = 0
"屏蔽诊断信息
let g:ycm_show_diagnostics_ui = 0
"跳转到定义
nnoremap <leader>jd :YcmCompleter GoToDefinitionElseDeclaration<CR>
" 输入第2个字符开始补全
let g:ycm_min_num_of_chars_for_completion=2
"在注释输入中也能补全
let g:ycm_complete_in_comments = 1
"在字符串输入中也能补全
let g:ycm_complete_in_strings = 1
" 修改对C函数的补全快捷键，默认是CTRL + space，修改为ALT + ;
let g:ycm_key_invoke_completion = '<M-;>'

"==============powerline 插件设置================
"vim有一个状态栏 加上powline则有两个状态栏
"设置powerline状态栏
set laststatus=2
let g:Powline_symbols='fancy'
set enc=utf-8
let termencoding=&encoding
set fileencodings=utf-8,gbk,ucs-bom,cp936
set guifont=Ubuntu\ Mono\ for\ Powerline\ 12

"==============NERDTree 插件设置=================
" 注释的时候自动加个空格, 强迫症必配
let g:NERDSpaceDelims=1

let NERDTreeMinimalUI=1
let NERDChristmasTree=1
" 是否显示隐藏文件
let NERDTreeShowHidden=1
" 显示书签列表
let NERDTreeShowBookmarks=0
" 显示行号
let NERDTreeShowLineNumbers=1
let NERDTreeAutoCenter=1
" 在终端启动vim时，共享NERDTree
let g:nerdtree_tabs_open_on_console_startup=0
" 忽略一下文件的显示
let NERDTreeIgnore=['\.pyc','\~$','\.swp']

let g:NERDTreeGitStatusIndicatorMapCustom = {
    \ "Modified"  : "✹",
    \ "Staged"    : "✚",
    \ "Untracked" : "✭",
    \ "Renamed"   : "➜",
    \ "Unmerged"  : "═",
    \ "Deleted"   : "✖",
    \ "Dirty"     : "✗",
    \ "Clean"     : "✔︎",
    \ "Unknown"   : "?"
    \ }
" Give a shortcut key to NERD Tree
map <F3> :NERDTreeMirror<CR>
map <F3> :NERDTreeToggle<CR>
let g:nerdtree_tabs_meaningful_tab_names=1
"=========clang-format=====================
"cltr+k 格式化 需要python支持
"map <C-K> :pyf <path-to-this-file>/clang-format.py<cr>
"imap <C-K> <c-o>:pyf <path-to-this-file>/clang-format.py<cr>
"==========ctags设置 Ctrl-F12=============
map <F12> :!ctags -R --c++-kinds=+p --fields=+iaS --extra=+q .<CR>

"================删除空白行 高亮空格======================
" show trailing spaace(s): http://vim.wikia.com/wiki/Highlight_unwanted_spaces
highlight ExtraWhitespace ctermbg=red guibg=red
match ExtraWhitespace /\s\+$/
autocmd BufEnter,WinEnter * match ExtraWhitespace /\s\+$/
autocmd InsertEnter * match ExtraWhitespace /\s\+\%#\@<!$/
autocmd InsertLeave * match ExtraWhitespace /\s\+$/

" auto remove trailing space(s) when saving
autocmd BufWritePre * :%s/\s\+$//e

