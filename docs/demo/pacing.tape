# Records docs/demo/pacing.gif from a real run: `vhs docs/demo/pacing.tape` from the repo root.
# Expects the Criteo Attribution prices and the CTR pool under ../.bench-data/ad-auction-lab.
Output docs/demo/pacing.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 300
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && source .venv/bin/activate && export D=../.bench-data/ad-auction-lab && clear` Enter
Sleep 3s
Show
Type `python -m ad_auction_lab pace --data $D --horizon 2000000 --advertisers 10 --budget-share 0.3` Enter
Wait+Line@400s /\$$/
Sleep 4s
