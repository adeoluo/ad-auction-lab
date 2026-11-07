# Records docs/demo/ctr.gif from a real run: `vhs docs/demo/ctr.tape` from the repo root.
# Expects Criteo prepared by `make data` under ../.bench-data/ad-auction-lab/criteo.
Output docs/demo/ctr.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 260
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && source .venv/bin/activate && export D=../.bench-data/ad-auction-lab && clear` Enter
Sleep 3s
Show
Type `python -m ad_auction_lab train-ctr --data $D --model fm --k 16 --lr 0.02 --train-rows 1000000 --test-rows 200000` Enter
Wait+Line@400s /\$$/
Sleep 4s
