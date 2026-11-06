# Records docs/demo/auction.gif from a real run: `vhs docs/demo/auction.tape` from the repo root.
# Needs no data: the incentive check draws values and quality scores itself.
Output docs/demo/auction.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 600
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && source .venv/bin/activate && export D=../.bench-data/ad-auction-lab && clear` Enter
Sleep 3s
Show
Type `python -m ad_auction_lab auction --bids 4,2,1 --quality 0.05,0.08,0.06 --slots 0.3,0.2 --reserve 0.5` Enter
Wait+Line@400s /\$$/
Sleep 2s
Type `python -m ad_auction_lab incentives --auctions 5000 --bidders 5` Enter
Wait+Line@400s /\$$/
Sleep 4s
