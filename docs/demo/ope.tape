# Records docs/demo/ope.gif from a real run: `vhs docs/demo/ope.tape` from the repo root.
# Expects the Open Bandit Dataset prepared by `make data` under ../.bench-data/ad-auction-lab/obd.
Output docs/demo/ope.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 330
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && source .venv/bin/activate && export D=../.bench-data/ad-auction-lab && clear` Enter
Sleep 3s
Show
Type `python -m ad_auction_lab ope --data $D --campaign all --target bts --boot 200` Enter
Wait+Line@400s /\$$/
Sleep 4s
