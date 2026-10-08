"""News events and who they are good or bad for - the part of "is this good news?" that wording alone gets wrong.

FinBERT reads tone. Tone breaks on exactly the news that moves stocks most:
  - "Merck to acquire Madrigal" sounds good for Merck; it's the company being BOUGHT (Madrigal) that jumps.
  - "Goldman Sachs downgrades Apple" is bad news for Apple, not for Goldman.
  - "Prices $500 million share offering" sounds positive; new shares dilute the owners (the stock usually drops).
  - "Beats estimates but cuts guidance" averages out; the guidance is what the market trades.
  - "Apple Q3 EPS $1.20 vs $1.35 est" has no tone words at all.
  - "Why Nvidia stock jumped today" describes a move that already happened.

This module spots those events with plain rules (no download, explainable), decides which company each one applies
to (the acquirer or the target, the analyst or the rated company, the winner or the one replaced), and raises
flags (recap, opinion, round-up, rumour, denial) that lower or remove confidence. engine.py combines the result
with FinBERT. Typical directions and strengths come from event studies (earnings surprises, guidance changes, M&A
targets, FDA decisions, offerings, buybacks, dividend cuts...).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------- text helpers
_SOURCE_SUFFIX = re.compile(r"\s+[-–—|]\s+(reuters|bloomberg|cnbc|wsj|marketwatch|benzinga|barron'?s|ft|"
                            r"financial times|ap|axios|the information|sources?|report|dow jones|nikkei)\s*$", re.I)
_CLAUSE_SPLIT = re.compile(r"\s*;\s*|\s+[-–—]\s+|,?\s+\b(?:but|while|although|though|however|whereas|despite|"
                           r"even as|yet)\b\s+", re.I)


_SPOKEN = re.compile(r"\b(?:f d a|s e c|e p s|d o j|f t c|c e o|c f o|i p o)\b", re.I)


def ascii_fold(text: str) -> str:
    """'Estée Lauder’s' -> "Estee Lauder's" (same length rules for every lookup)."""
    text = (text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # spoken acronyms in transcripts ("the f d a has approved"); padded so positions don't move
    return _SPOKEN.sub(lambda m: m.group(0).replace(" ", "") + "  ", text)


def clauses(text: str) -> list[str]:
    text = _SOURCE_SUFFIX.sub("", ascii_fold(text).strip())
    return [c.strip(" ,.") for c in _CLAUSE_SPLIT.split(text) if c and c.strip(" ,.")]


# --------------------------------------------------------------------------------------------- data classes
@dataclass
class Event:
    kind: str           # e.g. "guidance_cut"
    label: str          # plain English, shown in the app
    direction: int      # +1 good for the company, -1 bad, 0 = no clear direction (e.g. the acquirer in a deal)
    strength: float     # how reliably this kind of news moves the stock that way (0..1)
    role: str           # which company it applies to: subject | object | any | loser | actor_neutral
    evidence: str = ""  # the words that matched
    detail: str = ""    # e.g. "EPS $1.20 vs $1.35 expected (-11%)"
    horizon: str = "hours"
    priority: int = 0   # guidance (2) beats earnings (1) when one headline has both


@dataclass
class Reading:
    """What the rules make of one item for one company."""

    events: list[Event] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)  # recap, opinion, roundup, unconfirmed, denial, person_only
    direction: int = 0
    strength: float = 0.0
    neutral_reason: str = ""  # set when an event exists but says "no clear direction" (or a flag rules it out)

    @property
    def event(self) -> Event | None:
        return self.events[0] if self.events else None

    @property
    def label(self) -> str:
        return " + ".join(dict.fromkeys(e.label for e in self.events))


# --------------------------------------------------------------------------------------------- rules
# Each rule: (kind, label, direction, strength, role, pattern, priority, horizon)
# Roles: subject = the company named BEFORE the verb (or the only company), object = AFTER it,
#        any = whichever company is named, loser = the company after "over / replacing / outbidding",
#        actor_neutral = the company before the verb gets "neutral" (analyst houses, acquirers) and the one after
#        gets the direction, first = only the first company named, last = only the last one named (the others
#        are the ones acting).
_N = r"(?:\$?\d[\d,.]*\s*(?:%|percent|bln|billion|mln|million|bn|b|m|k)?)"
RULES: list[tuple] = [
    # ---- M&A ----
    ("mna_target", "Buyout offer (target)", +1, 0.92, "actor_neutral",
     r"\b(?:to (?:buy|acquire|purchase|take over|swallow)|(?:agrees?|agreed|plans?|offers?|offered|moves?|seeks?|"
     r"is set|in (?:advanced |early |exclusive )?talks|nears? (?:a )?deal|close to (?:a )?deal) to (?:buy|acquire|"
     r"purchase|take over)|buys|acquires|acquiring|(?:makes?|made|launches?|launched|submits?|sweetens?|raises?) "
     r"(?:an? |its )?(?:\$?[\d.,]+ ?(?:bln|billion|mln|million|bn|b|m)? )?(?:takeover |buyout |acquisition |cash |"
     r"all-cash |hostile |unsolicited )*(?:offer|bid|approach|proposal) for|bids? for|approached .{0,20}about a "
     r"(?:takeover|deal|merger)|takeover of|to take (?!it\b|the company\b)(?:[\w&'.-]+ ){1,5}?private)\b", 0,
     "immediate"),
    ("mna_target_it", "Buyout offer (target)", +1, 0.92, "subject",
     r"\b(?:to |plans? to |offer to )?(?:take|taking) (?:it|the company) private\b|\b(?:to|offer to) (?:buy|acquire) "
     r"(?:it|the company)\b", 0, "immediate"),
    ("mna_target_passive", "Buyout offer (target)", +1, 0.92, "subject",
     r"\b(?:to be (?:acquired|bought|taken private|taken over)|agrees? to be (?:acquired|bought|sold)|"
     r"(?:receives?|received|gets?|got|rejects?|rejected|weighs?|considers?) (?:an? |a new |a sweetened |another )?"
     r"(?:\$?[\d.,]+ ?(?:bln|billion|mln|million|bn|b|m)? )?(?:takeover|buyout|acquisition|purchase|cash) "
     r"(?:offer|bid|approach|proposal|interest)|explores? (?:a )?(?:sale|strategic alternatives|options including "
     r"a sale)|exploring (?:a )?sale|puts? itself up for sale|(?:shares? )?(?:jump|soar|surge)s? on (?:takeover|"
     r"buyout|deal) (?:talk|report)s?|takeover target|sells? itself)\b", 0, "immediate"),
    ("mna_collapse_active", "Deal fell apart", -1, 0.85, "actor_neutral",
     r"\b(?:terminates?|terminated|scraps?|scrapped|abandons?|abandoned|walks? away from|calls? off|called off|"
     r"pulls? out of|drops?|dropped|withdraws?|withdrew|ends?) (?:its |the |a )?(?:[\w$.,'&-]+\s+){0,4}?(?:merger|deal|acquisition|takeover|buyout|"
     r"bid|offer)\b", 0, "immediate"),
    ("mna_collapse", "Deal fell apart", -1, 0.85, "last",
     r"\b(?:deal|merger|talks|takeover|acquisition|bid) (?:(?!(?:that|which|would|will|to|could)\b)[\w&'.-]+ ){0,7}?"
     r"(?:collapses?|collapsed|falls? (?:apart|"
     r"through)|fell (?:apart|through)|terminated|blocked|scrapped|called off|ends?|ended)\b|\b(?:ftc|doj|regulators?|"
     r"court|judge) (?:[\w-]+ ){0,3}(?:blocks?|blocked|sues? to block)\b|\bsues? to block\b", 0, "immediate"),
    ("divestiture", "Selling a business", +1, 0.6, "subject",
     r"\b(?:to sell|sells?|selling|spin(?:s|ning)? off|spin-off of|carve[- ]out|divest\w*|explor\w+ (?:a )?sale of)"
     r" (?:its |the |a )?(?:[\w-]+ ){0,3}(?:unit|business|division|arm|segment|subsidiary)\b", 0, "hours"),
    ("licensing", "Licensing deal", +1, 0.66, "subject",
     r"\b(?:licens(?:e|es|ing) (?:its |the )?(?:[\w-]+ ){0,3}to|royalty deal|licensing (?:deal|agreement))\b", 0,
     "hours"),
    ("stake", "New stake / investment", +1, 0.68, "object",
     r"\b(?:to invest|invests?|invested|investing|takes? (?:an? )?(?:\$?[\d.,]+ ?(?:bln|billion|mln|million)? )?"
     r"stake|(?:discloses?|disclosed|reveals?|revealed|builds?|built) (?:an? )?(?:new )?(?:\$?[\d.,]+ ?(?:bln|"
     r"billion|mln|million)? )?(?:stake|position))\b", 0, "hours"),
    # ---- analysts (the firm doing the rating is neutral; the rated company gets the direction) ----
    ("analyst_upgrade", "Analyst upgrade", +1, 0.72, "actor_neutral",
     r"\b(?:upgrades?|(?:raises?|lifts?|moves?|bumps?|ups) .{0,40}? to (?:buy|outperform|overweight|strong buy|"
     r"positive|accumulate|add)|"
     r"initiates? .{0,40}?(?:with|at) (?:an? )?(?:buy|outperform|overweight|strong buy)|starts? .{0,30}?"
     r"(?:at|with) (?:buy|outperform|overweight)|adds? .{0,40}? to (?:its |the )?(?:conviction|top pick|"
     r"tactical outperform|outperform|focus)\w* list)\b", 0, "hours"),
    ("analyst_downgrade", "Analyst downgrade", -1, 0.74, "actor_neutral",
     r"\b(?:downgrades?|(?:cuts?|moves?|takes?|drops?) .{0,40}? to (?:sell|underperform|underweight|neutral|hold|"
     r"market perform|equal[- ]weight|"
     r"reduce)|lowers? .{0,40}? to (?:sell|underperform|underweight|neutral|hold)|initiates? .{0,40}?(?:with|at) "
     r"(?:an? )?(?:sell|underperform|underweight)|removes? .{0,40}? from (?:its |the )?\w* ?list)\b", 0, "hours"),
    ("analyst_upgrade_passive", "Analyst upgrade", +1, 0.72, "subject",
     r"\b(?:upgraded|raised to (?:buy|outperform|overweight)|initiated (?:at|with) (?:buy|outperform|overweight))\b",
     0, "hours"),
    ("analyst_downgrade_passive", "Analyst downgrade", -1, 0.74, "subject",
     r"\b(?:downgraded|cut to (?:sell|underperform|underweight|neutral|hold)|lowered to (?:sell|underperform|"
     r"underweight|neutral|hold))\b", 0, "hours"),
    ("target_raise", "Price target raised", +1, 0.6, "actor_neutral",
     r"\b(?:raises?|lifts?|boosts?|hikes?|increases?|bumps? up|ups) (?:.{0,40}?(?:price target|\bpt\b|target "
     r"price)\b|(?:[\w&'.-]+ ){1,3}target to \$?\d)",
     0, "hours"),
    ("target_cut", "Price target cut", -1, 0.62, "actor_neutral",
     r"\b(?:cuts?|lowers?|trims?|slashes?|reduces?) (?:.{0,40}?(?:price target|\bpt\b|target price)\b|"
     r"(?:[\w&'.-]+ ){1,3}target to \$?\d)", 0, "hours"),
    ("target_raise_passive", "Price target raised", +1, 0.6, "subject",
     r"\b(?:price target|\bpt\b|target price) (?:raised|lifted|boosted|hiked|increased|upped)\b", 0, "hours"),
    ("target_cut_passive", "Price target cut", -1, 0.62, "subject",
     r"\b(?:price target|\bpt\b|target price) (?:cut|lowered|trimmed|slashed|reduced)\b", 0, "hours"),
    # ---- guidance (beats the earnings result when both are in one headline) ----
    ("guidance_raise", "Raised its forecast", +1, 0.82, "subject",
     r"\b(?:raises?|raised|lifts?|lifted|boosts?|boosted|hikes?|hiked|increases?|increased|ups|upped|betters|"
     r"improves?) (?:its |the )?(?:full[- ]year |annual |fy\d* |fiscal(?:[- ]year)? |\d{4} |quarterly |q\d )*"
     r"(?:(?!(?:concerns?|questions?|doubts?|fears?|worries|alarm|hopes?|price|share|stock|pt|confidence|optimism|bets?|stakes?|expectations)\b)[\w&-]+ ){0,4}?"
     r"(?:guidance|outlook|forecast|view|targets?(?! to \$?\d| price)|estimates?)\b|\bguides? (?:above|ahead of|higher)|"
     r"(?:outlook|forecast|guidance) (?:above|tops|beats|ahead of) (?:estimates|expectations|consensus)|"
     r"(?:strong|upbeat|bullish|rosy|robust|better-than-expected|raised) (?:[\w-]+ ){0,2}(?:guidance|outlook|forecast)",
     2, "immediate"),
    ("guidance_cut", "Cut its forecast", -1, 0.86, "subject",
     r"\b(?:cuts?|lowers?|lowered|slashes?|slashed|trims?|trimmed|reduces?|reduced|downgrades? its|pares?|"
     r"warns? on|lowballs?) (?:its |the )?(?:full[- ]year |annual |fy\d* |fiscal(?:[- ]year)? |\d{4} |quarterly |"
     r"q\d |holiday[- ]quarter )*(?:(?!(?:concerns?|questions?|doubts?|fears?|worries|alarm|hopes?|price|share|stock|pt|confidence|optimism|bets?|stakes?|expectations)\b)[\w&-]+ ){0,4}?"
     r"(?:guidance|outlook|forecast|view|targets?(?! to \$?\d| price)|estimates?)\b|\bguides? (?:below|lower|under)|"
     r"(?:guidance|outlook|forecast) (?:is |was |came in |looks )?(?:light|weak|soft|below|disappointing)\b|"
     r"(?:outlook|forecast|guidance) (?:below|misses|short of|disappoints|underwhelms)|"
     r"(?:weak|soft|disappointing|downbeat|gloomy|cautious|bleak|lower[- ]than[- ]expected|tepid) (?:[\w-]+ ){0,2}"
     r"(?:guidance|outlook|forecast)|profit warning|warns? (?:of|on) (?:lower|weaker|slowing|falling)", 2,
     "immediate"),
    ("guidance_withdrawn", "Withdrew its forecast", -1, 0.75, "subject",
     r"\b(?:withdraws?|withdrew|pulls?|pulled|suspends?|suspended|scraps?) (?:its |the )?(?:[\w-]+ ){0,2}"
     r"(?:guidance|outlook|forecast)\b", 2, "immediate"),
    # ---- earnings ----
    ("earnings_beat", "Beat estimates", +1, 0.78, "subject",
     r"\b(?:beats?|beat|tops?|topped|tops|surpass(?:es|ed)?|exceeds?|exceeded|smash(?:es|ed)?|crush(?:es|ed)?|"
     r"trounces?|(?:blows?|sails?|surges?|jumps?|races?|zooms?|soars?) past|ahead of|above|better than|outpaces?|"
     r"outpaced) (?:the )?(?:\w+[- ]?){0,4}"
     r"(?:estimates?|expectations?|forecasts?|consensus|views?|the street|street (?:estimates|forecasts|view)|"
     r"analysts'? (?:estimates|expectations|forecasts))\b|\b(?:revenue|sales|profit|earnings|eps|results) "
     r"(?:above|ahead of|top|tops|beat|beats|surpass\w*) (?:consensus|estimates|expectations)\b|"
     r"\bbetter[- ]than[- ]expected (?:[\w-]+ ){0,2}(?:results|earnings|profit|revenue|sales|quarter)|"
     r"\brecord (?:quarterly |annual )?(?:revenue|sales|profit|earnings|deliveries|quarter)\b|"
     r"\b(?:beats?|tops?) on (?:the )?(?:[\w-]+ ){0,2}(?:profit|revenue|sales|earnings|eps|top|bottom|growth|margins?)\b",
     1, "immediate"),
    ("earnings_miss", "Missed estimates", -1, 0.80, "subject",
     r"\b(?:miss(?:es|ed)?|falls? short of|fell short of|lags?|lagged|trails?|trailed|below|under|"
     r"worse than|disappoints?) (?:the )?(?:\w+[- ]?){0,4}(?:estimates?|expectations?|forecasts?|consensus|views?|"
     r"the street|street (?:estimates|forecasts|view)|analysts'? (?:estimates|expectations|forecasts))\b|"
     r"\b(?:revenue|sales|profit|earnings|eps|results) (?:below|miss|misses|short of|disappoint\w*)\b|"
     r"\bworse[- ]than[- ]expected (?:[\w-]+ ){0,2}(?:results|earnings|profit|revenue|sales|loss|quarter)|"
     r"\b(?:profit|sales|revenue|earnings|deliveries) (?:falls?|fell|drops?|dropped|slumps?|slumped|plunges?|"
     r"plunged|declines?|declined|shrinks?|shrank|tumbles?|tumbled) (?:short|more than expected)|"
     r"\bmiss(?:es|ed)? on (?:the )?(?:[\w-]+ ){0,3}(?:profit|revenue|sales|earnings|eps|growth|margins?|subscribers|"
     r"deliveries|bookings)\b", 1, "immediate"),
    ("results_up", "Strong results", +1, 0.62, "subject",
     r"\b(?:revenue|sales|profit|earnings|bookings|net income|deliveries) (?:[\w-]+ ){0,2}(?:jumps?|jumped|surges?|surged|"
     r"soars?|soared|rises?|rose|grows?|grew|climbs?|climbed|more than doubles?|doubles?) (?:by )?\d+(?:\.\d+)?%|"
     r"\b(?:revenue|sales|profit|earnings) (?:[\w-]+ ){0,2}more than doubles?\b", 1, "hours"),
    ("results_down", "Weak results", -1, 0.62, "subject",
     r"\b(?:revenue|sales|profit|earnings|bookings|net income|deliveries) (?:[\w-]+ ){0,2}(?:falls?|fell|drops?|dropped|"
     r"declines?|declined|slumps?|slumped|plunges?|plunged|sinks?|sank|shrinks?|shrank|tumbles?|tumbled) (?:by )?"
     r"\d+(?:\.\d+)?%", 1, "hours"),
    ("slowdown", "Growth slowing", -1, 0.66, "subject",
     r"\b(?:growth|sales|demand|revenue|backlog|bookings|subscriber growth|user growth) (?:[\w-]+ ){0,2}(?:slows?|slowed|"
     r"slowing|decelerat\w+|weakens?|weakened|stalls?|stalled)\b", 1, "immediate"),
    # ---- FDA / clinical ----
    ("fda_approval", "FDA approval", +1, 0.75, "any",
     r"\b(?:fda|ema|regulators?|health canada|mhra|european commission)\b.{0,30}\b(?:approves?|approved|clears?|"
     r"cleared|grants? (?:accelerated |full |conditional )?approval|backs?|recommends? approval)\b|\b(?:wins?|won|"
     r"receives?|received|gets?|got|secures?|secured|earns?) (?:[\w-]+ ){0,3}(?:fda |ema |european )?(?:approval|"
     r"clearance|nod|green light|label(?: expansion)?|expanded (?:fda )?label)\b|\bbreakthrough (?:therapy )?designation\b"
     r"|\bfda (?:approval|nod|clearance) for\b", 0, "immediate"),
    ("fda_rejection", "FDA rejection / setback", -1, 0.88, "any",
     r"\b(?:complete response letter|\bcrl\b|refuse to file|clinical hold|fda (?:rejects?|rejected|declines?|"
     r"declined|denies?|denied|delays?|delayed)|(?:rejects?|rejected) (?:[\w-]+ ){0,3}(?:drug|therapy|application)|"
     r"advisory (?:committee|panel) (?:votes? (?:\d+[- ]to[- ]\d+ |\d+-\d+ )?)?against|(?:panel|adcom) (?:rejects|votes against))\b", 0,
     "immediate"),
    ("trial_fail", "Trial failed", -1, 0.88, "any",
     r"\b(?:fails?|failed|did not|didn't|does not|doesn't) to (?:improve|meet|show|extend|hit|reach|beat|slow|reduce)"
     r"\b|\bdisappoints? (?:on|in) (?:[\w-]+ ){0,3}(?:trial|study|phase|tolerability|efficacy|safety|data)\b|\b(?:fails?|failed|misses|missed|did not meet|didn't meet|does not meet|falls? short on) (?:its |the |a )?"
     r"(?:[\w-]+ ){0,2}(?:primary |main |key )?(?:endpoint|goal|trial|study)\b|\b(?:halts?|halted|stops?|stopped|"
     r"pauses?|paused|discontinues?|discontinued|terminates?|terminated) (?:its |the |a )?(?:\w+[- ]?){0,3}"
     r"(?:trial|study|program|development)\b|\bpatient death\b", 0, "immediate"),
    ("trial_success", "Trial success", +1, 0.80, "any",
     r"\b(?:meets?|met|hits?|achieves?|achieved|succeeds? on|reaches?|reached) (?:its |the |all |both )?"
     r"(?:[\w-]+ ){0,2}(?:primary |main |key )?(?:endpoints?|goals?)\b|\bpositive (?:topline |top-line |pivotal |"
     r"phase \d |late-stage )*(?:results|data|readout)\b|\bshows? (?:up to )?\d+(?:\.\d+)?% (?:weight loss|"
     r"reduction|improvement)|\b(?:extends?|extended|improves?|improved|prolongs?) (?:overall |progression-free )?"
     r"survival\b|\breduces? (?:the )?risk of (?:death|progression|heart attack|stroke)|\bshows? (?:strong|robust|"
     r"impressive|durable|positive) (?:[\w-]+ ){0,2}(?:response|efficacy|data|results)", 0, "immediate"),
    # ---- shares and payouts ----
    ("offering", "Share offering (dilution)", -1, 0.80, "any",
     r"\b(?:prices?|priced|launches?|launched|announces?|announced|files? for|plans?|proposes?|commences?|"
     r"upsizes?|upsized)? ?(?:an? |its |\$?[\d.,]+ ?(?:bln|billion|mln|million|bn|b|m)? )?(?:proposed |"
     r"underwritten |public |secondary |follow-on |registered direct |overnight |upsized |discounted )*"
     r"(?:(?:stock|share|equity|common stock) (?:offering|sale)|offering of (?:common )?(?:stock|shares)|"
     r"(?<!initial )(?<!initial-)(?:secondary |follow-on |underwritten )?public offering|"
     r"at-the-market (?:offering|program|equity program)|\batm (?:offering|program)\b|(?:convertible|exchangeable) (?:senior )?"
     r"(?:notes?|bonds?|offering|debt)|private placement|dilutive|dilution)\b|\bsell(?:s|ing)? (?:up to )?\$?[\d.,]+ "
     r"?(?:bln|billion|mln|million|bn|b|m)? (?:in|of|worth of) (?:new )?(?:class [a-c] )?(?:common )?(?:stock|shares)\b",
     0, "immediate"),
    ("buyback", "Share buyback", +1, 0.70, "any",
     r"\b(?:buyback|share repurchase|stock repurchase|repurchase (?:program|plan|authorization)|buy back)\b", 0,
     "hours"),
    ("dividend_cut", "Dividend cut", -1, 0.84, "any",
     r"\b(?:cuts?|slashes?|slashed|halves?|halved|suspends?|suspended|eliminates?|eliminated|omits?|scraps?|"
     r"reduces?|reduced|lowers?) (?:its |the )?(?:quarterly |annual |interim |monthly |final )?"
     r"(?:dividend|payout|distribution)s?\b|\bdividend (?:cut|suspension|slashed|halved)\b", 0, "immediate"),
    ("dividend_raise", "Dividend raised", +1, 0.6, "any",
     r"\b(?:raises?|raised|hikes?|hiked|increases?|increased|boosts?|boosted|lifts?) (?:its |the )?(?:quarterly |"
     r"annual |interim |monthly |final )?(?:dividend|payout|distribution)\b|\b(?:initiates?|declares?|announces?|"
     r"pays?) (?:an? |its )?(?:\$?[\d.,]+(?: per share)? )?(?:first|first-ever|inaugural|special|one-time|extra)\b "
     r"(?:[\w-]+ )?dividend\b|"
     r"\binitiates? (?:an? |a quarterly )?dividend\b|\bbigger-than-expected dividend\b", 0,
     "hours"),
    ("split", "Stock split", +1, 0.6, "any", r"\b\d+[- ]for[- ]\d+ (?:forward )?stock split\b|\bannounces? (?:a )?"
     r"stock split\b", 0, "hours"),
    ("reverse_split", "Reverse stock split", -1, 0.7, "any", r"\breverse (?:stock )?split\b", 0, "hours"),
    # ---- index membership ----
    ("index_add", "Joining a major index", +1, 0.72, "subject",
     r"\b(?:to join|joins?|joining|added to|to be added to|set to join|will join|enters?|to enter) (?:the )?"
     r"(?:s&p 500|s&p500|nasdaq[- ]100|dow jones industrial average|dow|russell 1000|s&p midcap 400)\b", 0, "hours"),
    ("index_remove", "Removed from a major index", -1, 0.72, "subject",
     r"\b(?:removed from|to be removed from|dropped from|to leave|leaves|deleted from|kicked out of) (?:the )?"
     r"(?:s&p 500|s&p500|nasdaq[- ]100|dow jones industrial average|dow)\b|\b(?:passed over|snubbed|left out|"
     r"overlooked) (?:again )?(?:for|of) (?:[\w&]+ ){0,3}(?:inclusion|index)", 0, "hours"),
    # ---- trouble ----
    ("bankruptcy", "Bankruptcy / default risk", -1, 0.92, "any",
     r"\b(?:files? for|filed for|filing for|prepares? (?:to file )?for|nears?|considers?) (?:chapter 11|"
     r"bankruptcy)|\bchapter 11\b|\bgoing[- ]concern\b|\bdefaults? on\b|\bmissed (?:a |an )?(?:interest |"
     r"debt |bond )?payment\b|\brestructuring advis[eo]rs?\b|\bdelisting (?:notice|warning)\b|"
     r"\b(?:may |could )?faces? (?:\w+ )?delisting\b",
     0, "immediate"),
    ("accounting", "Accounting problem", -1, 0.84, "any",
     r"\b(?:restates?|restated|restatement|accounting (?:irregularities|errors?|problems?|issues|probe)|auditor "
     r"(?:resigns?|quits?)|material weakness|delays? (?:its )?(?:annual|quarterly) (?:report|filing))\b", 0,
     "immediate"),
    ("short_report", "Short-seller report", -1, 0.78, "any",
     r"\b(?:short[- ]seller|short report|hindenburg|muddy waters|citron|spruce point|grizzly research|"
     r"wolfpack|culper|fuzzy panda|kerrisdale) (?:[\w-]+ ){0,3}(?:report|targets?|alleges?|accuses?|bets? against|"
     r"says|discloses? short)|\bshort[- ]seller\b", 0, "immediate"),
    ("legal", "Probe / lawsuit", -1, 0.70, "any",
     r"\b(?:probes?|probed|probing|investigat\w+|subpoena\w*|indicted|indictment|charged? with|charges against|"
     r"sues|sued|lawsuit|class action|antitrust (?:suit|lawsuit|case|probe)|fined|fines|fine of|penalty|verdict|"
     r"hit with (?:an? )?(?:\$?[\d.,]+ ?(?:bln|billion|mln|million)? )?(?:\w+ )?(?:suit|lawsuit|fine|penalty|probe)|"
     r"jury (?:orders?|finds?|awards?)|verdict against|ordered to pay|raid(?:ed|s)? (?:\w+ )?offices?)\b|"
     r"\b(?:sec|doj|ftc|justice department|attorneys? general|prosecutors?|regulators?) (?:[\w-]+ ){0,3}"
     r"(?:probe|investigation|charges|sues|accuses)\b", 0, "hours"),
    ("ceo_exit", "CEO leaving", -1, 0.62, "any",
     r"\b(?:ceo|chief executive|cfo|chief financial officer|founder)\b (?:[\w-]+ ){0,3}(?:resigns?|resigned|steps? "
     r"down|stepping down|to step down|quits?|departs?|exits?|ousted|fired|out\b|leaves|leaving|is out)|"
     r"\b(?:ousts?|ousted|fires?|fired) (?:its |the )?(?:ceo|chief executive)\b", 0, "immediate"),
    ("recall", "Recall / safety problem", -1, 0.62, "any",
     r"\b(?:recalls?|recalled|recalling|grounds?|grounded|grounding|blows? out|crash(?:es|ed)?|explosion|"
     r"fire at|contamination|outbreak|e\. ?coli|salmonella|listeria|food poisoning|safety (?:probe|investigation|"
     r"warning))\b", 0, "hours"),
    ("delay", "Launch delayed", -1, 0.62, "any",
     r"\b(?:launch|rollout|release|debut|production|deliveries|approval|start) (?:[\w-]+ ){0,2}(?:delayed|pushed back|"
     r"postponed|slips?)\b|\bdelays? (?:the |its )?(?:[\w-]+ ){0,2}(?:launch|rollout|release|debut)\b", 0, "hours"),
    ("activist", "Activist investor stake", +1, 0.7, "any",
     r"\bactivist (?:investor |hedge fund |fund |shareholder )?(?:[\w-]+ ){0,3}(?:builds?|building|takes?|taking|has|"
     r"holds?|discloses?|disclosed|amass\w*|buys?|bought|acquires?|acquired) (?:an? )?(?:[\w-]+ ){0,2}(?:stake|position)|"
     r"\bactivist stake\b", 0, "hours"),
    ("breach", "Hack / outage", -1, 0.6, "any",
     r"\b(?:hack(?:ed|ers?)?|data breach|breach|cyberattack|cyber attack|ransomware|outage|stole|stolen|"
     r"leaked)\b", 0, "hours"),
    ("halt", "Production halt / strike", -1, 0.66, "any",
     r"\b(?:halts?|halted|halting|suspends?|suspended|stops?|stopped|pauses?|paused) (?:[\w-]+ ){0,2}"
     r"(?:production|output|operations|shipments|deliveries|sales)\b|\bproduction halt\b|\bon strike\b|"
     r"\bstrike (?:begins|starts|continues|widens)\b|\bwalk(?:s|ed)? off the job\b", 0, "hours"),
    ("strike_end", "Strike ends / labour deal", +1, 0.65, "any",
     r"\b(?:tentative (?:labor |labour )?(?:deal|agreement|contract)|end(?:s|ed|ing)? (?:the |a )?strike|strike "
     r"ends|ratif(?:y|ies|ied) (?:\w+ )?(?:contract|deal))\b", 0, "hours"),
    ("trade_hit", "Tariff / export ban", -1, 0.62, "any",
     r"\b(?:export (?:ban|curbs?|controls?|restrictions?)|banned from|blacklist(?:s|ed)?|entity list|sanctions? "
     r"on|tariffs? (?:on|hit|hits|hurt|weigh))\b", 0, "hours"),
    # ---- business wins ----
    ("contract", "Contract / deal win", +1, 0.72, "subject",
     r"\b(?:wins?|won|lands?|landed|secures?|secured|awarded|gets?|receives?|signs?|signed|clinches?|bags?|nabs?|"
     r"inks?|(?:selected|chosen|picked|tapped) for) (?:an? |the |its |[\w&.-]+'s )?(?:\$?[\d.,]+ ?(?:bln|billion|mln|"
     r"million|bn|b|m)?[- ]?)?(?:[\w&.'-]+[- ]?){0,5}?"
     r"(?:contract|order|orders|deal|award|program|programme|rights|games|account|tender|business)\b|\bnamed (?:the )?"
     r"(?:official|exclusive|preferred|primary) (?:[\w-]+ ){0,3}(?:provider|partner|supplier|sponsor)\b", 0, "hours"),
    ("chosen", "Picked by a customer", +1, 0.70, "object",
     r"\b(?:picks?|picked|chooses?|chose|selects?|selected|taps?|tapped|hires?|hired|switches? to|switched to)\b",
     0, "hours"),
    ("partnership", "Partnership", +1, 0.6, "first",
     r"\b(?:partners? with|partnership|teams? up|collaborat\w+|alliance|joint venture|expand(?:s|ed)? (?:\w+ )?"
     r"partnership|to partner)\b", 0, "hours"),
    ("layoffs", "Job cuts / restructuring", 0, 0.0, "any",
     r"\b(?:layoffs?|lay off|laying off|job cuts|cuts? (?:about |around |nearly |some )?[\d,]+ (?:[\w-]+ ){0,2}(?:jobs|"
     r"positions|roles|workers|employees|staff)|to cut (?:[\w-]+ ){0,3}jobs|restructuring|workforce reduction|"
     r"reduce (?:its )?workforce)\b", 0, "hours"),
    ("exec_named", "New executive", 0, 0.0, "any",
     r"\b(?:names?|named|appoints?|appointed|hires?|taps?) (?:[\w-]+ ){0,4}(?:as )?(?:new )?(?:ceo|cfo|chief|"
     r"president|chair\w*|director|successor)\b", 0, "hours"),
    ("routine", "Routine announcement", 0, 0.0, "any",
     r"\b(?:to (?:hold|host|present|participate|webcast)|will (?:hold|host|present|participate)|annual (?:general |"
     r"shareholders' |shareholder )?meeting|investor day|conference call|fireside chat|to report (?:[\w-]+ ){0,3}"
     r"results on|declares? (?:a |its )?(?:regular )?(?:quarterly |monthly |semi-annual |annual )?(?:cash )?"
     r"dividend)\b", 0, "hours"),
]
_COMPILED = [(k, lab, d, s, role, re.compile(p, re.I), pri, hz) for k, lab, d, s, role, p, pri, hz in RULES]

# "picks AMD over Nvidia", "replacing Capital One", "outbidding Disney's ESPN"
_LOSER = re.compile(r"\b(?:over|replacing|replaces|replaced|instead of|outbid(?:ding|s)?|beating out|beats out|"
                    r"at the expense of|ousting|displacing|displaces|unseating|wins? (?:\w+ )?(?:from|away from))\s",
                    re.I)

_SIZED = {"contract", "stake", "buyback"}
_BILLIONS = re.compile(r"\d\s*(?:bln|billion|bn|b)\b", re.I)

# ---- flags ----
_MOVES = (r"(?:rose|fell|jumped|tumbled|surged|plunged|climbed|dropped|slid|sank|gained|lost|rallied|slumped|soared|"
          r"skidded|slipped|crashed|doubled|tripled|halved)")
# a story about a move that already happened ("Why Nvidia stock jumped", "Shares of Boeing were down 4% - here's
# what drove the decline") -> not news to trade
_RECAP = re.compile(
    r"^why\b|^here'?s why\b|\bhere'?s (?:why|what)\b|\bwhat'?s (?:going on|happening|behind)\b|\bwhat (?:investors|"
    r"you) (?:need|should|want) to know\b|\bwhat (?:drove|happened|to know|caused)\b|\b(?:stocks?|shares) "
    r"(?:making|moving) (?:the )?(?:biggest )?moves\b|\b(?:biggest|top) (?:stock )?(?:movers|gainers|losers)\b|"
    r"\b(?:midday|mid-day|premarket|pre-market|after-hours|after hours|morning|afternoon) (?:movers|gainers|losers|"
    r"decliners)\b|\bstocks to watch\b|\b(?:closes?|closed|ended|finished|ends?) (?:up|down|higher|lower)\b|"
    r"\b(?:best|worst) (?:day|week|month)\b|\b" + _MOVES + r"\s+\d+(?:\.\d+)?%\s+(?:on |in )?(?:monday|tuesday|"
    r"wednesday|thursday|friday|yesterday|last week|in (?:early|late|morning|afternoon|extended|midday) trading|on the "
    r"day|this week|this year|so far)\b|\b(?:were|was) (?:up|down) \d+(?:\.\d+)?%|\b(?:monday|tuesday|wednesday|"
    r"thursday|friday|yesterday|last week)\b(?:\W+\w+){0,3}\W+(?:extending|after|following) (?:post-earnings|its|"
    r"the)\b|\bextend(?:s|ing)? (?:post-earnings |its |a )?(?:losses|gains|rally|slide|winning streak|losing streak)\b|"
    r"\bfor a (?:second|third|fourth|fifth|\d+(?:st|nd|rd|th)) (?:straight |consecutive )?(?:session|day|week)\b|"
    r"\b(?:has|have) (?:doubled|tripled|halved|soared|surged|plunged|crashed|rallied) (?:\w+ ){0,2}since\b|"
    r"\b" + _MOVES + r" \d+(?:\.\d+)?% yesterday\b", re.I)
# the share move named next to the news ("PayPal misses on checkout growth; shares fall") - the news still counts,
# but with no news event the item only describes a move
_MOVED = re.compile(
    r"\b(?:shares?|stocks?|the stock)\b(?:\s+\w+){0,3}\s+(?:" + _MOVES[3:-1] + r"|closed|ended|finished)\b|"
    r"\b(?:shares|stock) (?:is |are |was |were )?(?:soaring|surging|jumping|plunging|sinking|tumbling|falling|rising|"
    r"sliding|rallying|moving|trading (?:higher|lower)|up|down)\b|\bshares (?:jump|soar|surge|plunge|tumble|sink|fall|"
    r"drop|rise|rally|slide|climb|spike|crater|gain|slip|sag)s?\b|\b(?:moving|trading) (?:higher|lower)\b", re.I)
_OPINION = re.compile(
    r"^\d+ (?:reasons?|stocks?|things|ways|charts?|growth stocks|dividend stocks|ai stocks)\b|\b(?:is|are) (?:it|they|"
    r"\w+(?: \w+)?(?: stock)?) (?:a|still a|now a) (?:buy|sell|bargain|steal)\b|\bshould you (?:buy|sell|own|worry)\b|"
    r"\bbetter (?:buy|stock|investment)\b|\bstocks? to (?:buy|watch|avoid|sell|own)\b|\bmillionaire[- ]maker\b|"
    r"\bcould (?:soar|double|triple|skyrocket|crash|make you)\b|\bno[- ]brainer\b|\bbuy and hold\b|"
    r"\b(?:time to buy|buy the dip|too late to buy|before it'?s too late)\b|\bhere'?s the best\b|\bmy (?:favorite|top)\b|"
    r"\b(?:i|we) (?:think|believe|bought|sold)\b|\bsmartest\b|\bforever stock\b|\bshould you (?:sell|buy)\b",
    re.I)
_OPINION_URL = re.compile(r"fool\.com/(?:investing|stock-market)|seekingalpha\.com/article|/opinion/|"
                          r"investorplace\.com|zacks\.com/stock/news|247wallst\.com", re.I)
_RUMOR = re.compile(r"\b(?:reportedly|report says|reports say|according to (?:a |the )?(?:report|people|sources|"
                    r"media)|(?:sources|people) (?:say|said|familiar)|- sources|people familiar|in (?:early |advanced |"
                    r"preliminary )?talks|explor(?:es|ing|e)|consider(?:s|ing)|weighs|weighing|mulls|mulling|said to|"
                    r"is said|rumou?rs?|speculation|chatter|could|says (?:\w+ )?(?:analyst|kuo))\b|"
                    r"\s[-\u2013\u2014]\s*(?:sources|report|media report)\s*$", re.I | re.M)
_DENIAL = re.compile(r"\b(?:denies|denied|deny|refutes?|refuted|rebuts?|dismisses (?:\w+ )?(?:report|rumou?r)s?|"
                     r"(?:says|said) (?:it )?(?:is|was|has) not\b|(?:has|have) no (?:current |present |immediate )?(?:plans?|"
                     r"intention|intentions) (?:to|of)|not in talks|(?:is|are) not (?:pursuing|considering)|"
                     r"no (?:talks|discussions) (?:with|about)|false report|categorically)\b", re.I)
_QUESTION = re.compile(r"\?\s*$")
# press releases from law firms looking for clients ("INVESTOR ALERT: ... investigating XYZ on behalf of investors")
_LAW_FIRM_AD = re.compile(
    r"\b(?:investor alert|shareholder alert|investor notice|shareholder notice|lead plaintiff deadline|class action "
    r"(?:lawsuit )?(?:filed|reminder|deadline|alert)|on behalf of (?:investors|shareholders)|law firm|"
    r"rosen law|pomerantz|levi & korsinsky|bronstein|gewirtz|glancy|kessler topaz|faruqi|bragar eagel|schall law|"
    r"robbins geller|hagens berman|portnoy law|block & leviton|bernstein liebhard|gross law firm|kirby mcinerney|"
    r"holzer & holzer|johnson fistel|kahn swick|halper sadeh|ademi|brodsky & smith|monteverde|rigrodsky)\b", re.I)
_ROUTINE_DIVIDEND = re.compile(r"\b\d+(?:st|nd|rd|th) (?:(?:consecutive|straight|annual|yearly) ){0,2}(?:time|year|quarter|"
                               r"increase|raise|hike)\b|\b(?:consecutive|straight) (?:years?|quarters?)\b|"
                               r"\bdividend (?:king|aristocrat)", re.I)

# --------------------------------------------------------------------------------------------- numbers
_NUM = r"(-?\$?-?\(?\d+(?:,\d{3})*(?:\.\d+)?\)?)"
_UNIT = r"\s*(b|bn|bln|billion|m|mn|mln|million|k)?\b"
_VS = r"\s*(?:vs\.?|versus|v\.|compared (?:with|to)|against|,? (?:topping|beating|missing|above|below|ahead of))\s*"
_EST = r"\s*(?:est\.?|estimates?|consensus|expected|exp\.?|forecast|street|fcst|analysts?'? (?:estimate|forecast)s?|e\b)"
_EPS_VS = re.compile(r"\b(?:adj(?:usted)?\.?\s*|non-gaap\s*|gaap\s*|diluted\s*)?(?:eps|earnings per share|loss per "
                     r"share)\s*(?:of|was|came in at|:)?\s*" + _NUM + _VS + _NUM + _EST, re.I)
_REV_VS = re.compile(r"\b(?:rev(?:enue)?s?|sales|net sales|total revenue)\s*(?:of|was|came in at|:)?\s*" + _NUM +
                     _UNIT + _VS + _NUM + _UNIT + _EST, re.I)
_BY = re.compile(r"\b(beats?|tops?|misses?|missed|beat|topped)\b[^.;]{0,40}?\bby \$?(\d+(?:\.\d+)?)", re.I)
_PER_SHARE_FORECAST = re.compile(r"(-?\$?\d+(?:\.\d+)?) (?:per|a) share[^.;]{0,60}?(?:topping|beating|above|missing|"
                                 r"below|short of|versus|vs\.?|compared with) (?:the |analysts'? |wall street'?s? )?"
                                 r"(?:average |consensus )?(?:estimate |forecast |expectation )?(?:of )?"
                                 r"(-?\$?\d+(?:\.\d+)?)", re.I)


# "comparable sales -1.9% vs -1.5% est", "organic revenue +1.3% vs +2.5% est" (retailers / consumer goods trade on these)
_PCT_VS = re.compile(r"\b(?:comparable(?: store)? sales|comp(?:arable)?s?(?: sales)?|same[- ]store sales|organic "
                     r"(?:revenue|sales)|like[- ]for[- ]like sales)(?: growth)?\s*(?:of|was|were|came in at|:)?\s*"
                     r"([+-]?\d+(?:\.\d+)?)%" + _VS + r"([+-]?\d+(?:\.\d+)?)%" + _EST, re.I)
# "Sees Q4 revenue $1.40B-$1.50B vs $1.61B est" - a forecast against what analysts expect
_GUIDE_VS = re.compile(r"\b(?:sees|expects|forecasts?|guides?|projects?|outlook|guidance)\b[^.;]{0,40}?\b(?:rev(?:enue)?s?|"
                       r"sales|eps|earnings per share)\b[^.;$\d-]{0,15}" + _NUM + _UNIT + r"(?:\s*(?:-|to)\s*" + _NUM +
                       _UNIT + r")?" + _VS + _NUM + _UNIT + _EST, re.I)


def guidance_surprise(text: str) -> Event | None:
    """'Sees Q4 revenue $1.40B-$1.50B vs $1.61B est' -> forecast 10% below estimates."""
    m = _GUIDE_VS.search(ascii_fold(text))
    if not m:
        return None
    lo, hi, est = _num(m.group(1)), _num(m.group(3)) if m.group(3) else None, _num(m.group(5))
    if lo is None or est is None or est == 0:
        return None
    unit = m.group(2) or m.group(4)
    mid = (lo + (hi if hi is not None else lo)) / 2 * _scale(unit)
    est_v = est * _scale(m.group(6) or unit)
    sc = (mid - est_v) / abs(est_v)
    if abs(sc) < 0.01:
        return Event("guidance_inline", "Forecast in line", 0, 0.0, "subject", m.group(0), "", "immediate", 2)
    up = sc > 0
    return Event("guidance_raise" if up else "guidance_cut", "Forecast above estimates" if up else
                 "Forecast below estimates", 1 if up else -1, min(0.9, 0.76 + 2.0 * abs(sc)), "subject", m.group(0),
                 f"forecast midpoint {sc:+.1%} vs estimates", "immediate", 2)


def _num(text: str) -> float | None:
    t = text.replace("$", "").replace(",", "").strip()
    neg = t.startswith("-") or (t.startswith("(") and t.endswith(")"))
    t = t.strip("-()")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def _scale(unit: str | None) -> float:
    u = (unit or "").lower()
    return {"b": 1e9, "bn": 1e9, "bln": 1e9, "billion": 1e9, "m": 1e6, "mn": 1e6, "mln": 1e6, "million": 1e6,
            "k": 1e3}.get(u, 1.0)


def number_surprise(text: str) -> Event | None:
    """'EPS $1.20 vs $1.35 est; revenue $85B vs $89B est' -> a miss of 11% / 4.5%."""
    t = ascii_fold(text)
    g = _GUIDE_VS.search(t)
    if g:  # the forecast numbers are read by guidance_surprise, not as this quarter's results
        t = t[:g.start()] + " " * (g.end() - g.start()) + t[g.end():]
    parts, details = [], []
    m = _EPS_VS.search(t)
    if m:
        a, e = _num(m.group(1)), _num(m.group(2))
        if a is not None and e is not None and e != 0:
            s = (a - e) / abs(e)
            parts.append((0.6, s))
            details.append(f"EPS {m.group(1).strip()} vs {m.group(2).strip()} expected ({s:+.0%})")
    else:
        m = _PER_SHARE_FORECAST.search(t)
        if m:
            a, e = _num(m.group(1)), _num(m.group(2))
            if a is not None and e is not None and e != 0:
                s = (a - e) / abs(e)
                parts.append((0.6, s))
                details.append(f"EPS {m.group(1)} vs {m.group(2)} expected ({s:+.0%})")
    m = _REV_VS.search(t)
    if m:
        a, e = _num(m.group(1)), _num(m.group(3))
        if a is not None and e is not None and e != 0:
            a, e = a * _scale(m.group(2)), e * _scale(m.group(4) or m.group(2))
            s = (a - e) / abs(e)
            parts.append((0.4, s))
            details.append(f"revenue {m.group(1).strip()}{m.group(2) or ''} vs {m.group(3).strip()}"
                           f"{m.group(4) or m.group(2) or ''} expected ({s:+.1%})")
    m = _PCT_VS.search(t)
    if m:
        a, e = float(m.group(1)), float(m.group(2))
        s = (a - e) / (abs(e) + 2.0)  # percentage points, scaled so a 1-point miss on a small number counts
        parts.append((0.6, s))
        details.append(f"same-store/organic sales {a:+g}% vs {e:+g}% expected")
    if not parts:
        m = _BY.search(t)
        if m:
            beat = m.group(1).lower().startswith(("beat", "top"))
            return Event("earnings_beat" if beat else "earnings_miss", "Beat estimates" if beat else "Missed estimates",
                         1 if beat else -1, 0.78, "any", m.group(0), "", "immediate", 1)
        return None
    score = sum(w * max(-0.5, min(0.5, s)) for w, s in parts) / sum(w for w, _ in parts)
    if abs(score) < 0.005:
        return Event("earnings_inline", "In line with estimates", 0, 0.0, "any", "", "; ".join(details), "immediate", 1)
    direction = 1 if score > 0 else -1
    strength = min(0.88, 0.74 + 2.0 * abs(score))
    return Event("earnings_beat" if direction > 0 else "earnings_miss",
                 "Beat estimates" if direction > 0 else "Missed estimates", direction, strength, "any", "",
                 "; ".join(details), "immediate", 1)


# --------------------------------------------------------------------------------------------- reading
def read_events(text: str, target_spans: dict[int, list[tuple[int, int]]], url: str = "",
                actors: dict[int, tuple[str, ...]] | None = None) -> dict[int, Reading]:
    """Events for every company in one piece of text.

    target_spans: {company_index: [(start, end), ...]} - where each company is named in ascii_fold(text).
    actors: {company_index: kind prefixes} - companies that are always the ones acting for those kinds of event
            (an analyst house for "analyst"/"target", a buyout firm for "mna").
    Returns {company_index: Reading}. Spans must be computed on ascii_fold(text) (see find_spans).
    """
    actors = actors or {}
    folded = ascii_fold(text)
    out = {i: Reading() for i in target_spans}
    whole = folded.split("\n", 1)[0]
    # ---- whole-item flags ----
    flags: list[str] = []
    if _RECAP.search(whole):
        flags.append("recap")
    if _OPINION.search(whole) or (url and _OPINION_URL.search(url)):
        flags.append("opinion")
    elif _QUESTION.search(whole) and not re.search(r"\b(?:says|said|asks?|told)\b", whole, re.I):
        flags.append("opinion")
    if len(target_spans) >= 5:
        flags.append("roundup")
    if _RUMOR.search(folded[:400]):
        flags.append("unconfirmed")
    if _LAW_FIRM_AD.search(folded[:600]):
        flags.append("law_firm_ad")
    for r in out.values():
        r.flags = list(flags)

    # ---- per clause events, with roles ----
    last_subjects: list[int] = []
    for clause in _clause_spans(folded):
        c_start, c_end = clause
        ctext = folded[c_start:c_end]
        here = {i: [a for a, b in sp if c_start <= a < c_end
                    and not _DESCRIPTOR.match(folded, b)]  # "Nvidia partner Super Micro": about Super Micro
                for i, sp in target_spans.items()}
        first_span = {i: min((a, b) for a, b in target_spans[i] if a in ps) for i, ps in here.items() if ps}
        present = sorted((i for i, ps in here.items() if ps), key=lambda i: first_span[i])
        denial = bool(_DENIAL.search(ctext))
        found_any = False
        for kind, label, d, s, role, rx, pri, hz in _COMPILED:
            forced = [i for i in present if any(kind.startswith(k) for k in actors.get(i, ()))]
            others = [i for i in present if i not in forced]
            for m in rx.finditer(ctext):
                # "fails to improve survival", "is not in talks to acquire": read as the opposite
                neg = denial or bool(d > 0 and _NEGATED_BEFORE.search(ctext[:m.start()]))
                if kind == "mna_target" and m.group(0).lower().strip() == "to buy" and _RATING_BEFORE.search(
                        ctext[:m.start()]):
                    continue  # "upgrades Coinbase to Buy" is a rating, not a takeover
                verb_at = c_start + m.start()
                lead_words = bool(re.search(r"\w", re.sub(r"^\W*(?:and|but|while|as)?\W*", "", ctext[:m.start()],
                                                          flags=re.I)))
                before = [i for i in others if first_span[i][0] < verb_at]
                after = [i for i in others if i not in before]
                # the company (or "A and B") right after the verb - not one named later in the sentence
                near = _chained(after, first_span, folded)
                if not others:
                    # "...; raises outlook" -> same company as before (not "...; Saudi fund to buy a stake")
                    applies = ({i: role for i in last_subjects if i not in forced}
                               if role in ("any", "subject", "first", "last") and not forced else {})
                else:
                    applies = {}
                    if role == "any":
                        applies = {i: "any" for i in others}
                    elif role == "subject":
                        if before:
                            applies = {i: "subject" for i in before}
                        elif not lead_words and last_subjects:  # "..., but forecast disappoints as Apple ..."
                            applies = {i: "subject" for i in last_subjects if i not in forced}
                        elif not lead_words:
                            applies = {i: "subject" for i in others}
                        # else someone else is the subject ("TSMC wins Nvidia's orders" when TSMC isn't listed)
                    elif role == "first":
                        applies = {others[0]: "first"}
                    elif role == "last":
                        applies = {i: "actor" for i in others[:-1]}
                        applies[others[-1]] = "last"
                    elif role == "object":
                        applies = {i: "object" for i in near}
                        for i in before:
                            applies.setdefault(i, "actor")
                    elif role == "actor_neutral":
                        if near:
                            applies = {i: "object" for i in near}
                            for i in before:
                                applies[i] = "actor"
                        elif kind.startswith("mna"):  # "Apple in talks to acquire Perplexity" - Apple is the buyer
                            applies = {i: "actor" for i in before}
                        else:  # "Goldman upgrades Coinbase to Buy, raises PT" - the company nearest the verb
                            applies = {i: "actor" for i in before[:-1]}
                            if before:
                                applies[before[-1]] = "object"
                for i in forced:
                    applies[i] = "actor"
                for i, r_role in applies.items():
                    if r_role == "actor":
                        ev = Event(kind, label, 0, 0.0, "actor", m.group(0), "", hz, pri)
                    else:
                        big = kind in _SIZED and _BILLIONS.search(m.group(0))  # a $1bn+ contract matters more
                        ev = Event(kind, label, -d if neg and d else d, min(0.9, s + 0.08) if big else s, r_role,
                                   m.group(0), "", hz, pri)
                    if neg and d:
                        ev.label = f"Denied: {label.lower()}"
                    if kind == "dividend_raise" and _ROUTINE_DIVIDEND.search(ctext):
                        ev.direction, ev.strength, ev.label = 0, 0.0, "Regular dividend increase"
                    out[i].events.append(ev)
                found_any = found_any or bool(applies)
                break  # one match per rule per clause is enough
        # the company replaced / outbid / passed over is the loser of a win
        lm = _LOSER.search(ctext)
        if lm:
            loser_at = c_start + lm.end()
            for i in present:
                if min(here[i]) >= loser_at and not any(e.direction < 0 for e in out[i].events):
                    out[i].events = [e for e in out[i].events if e.role != "object"]
                    out[i].events.append(Event("lost_out", "Lost out to a rival", -1, 0.65, "loser", lm.group(0),
                                               "", "hours"))
        if _MOVED.search(ctext) and "recap" not in flags:  # "...; shares fall" - whose shares: this clause's company
            for i in (others or last_subjects):
                if "moved" not in out[i].flags:
                    out[i].flags.append("moved")
        cm = _COMPLAINANT.search(ctext)
        if cm:
            for i in present:
                if first_span[i][0] >= c_start + cm.end():
                    out[i].events = [e for e in out[i].events if e.kind != "legal"]
        if denial and not found_any:
            for i in present:
                if "denial" not in out[i].flags:
                    out[i].flags.append("denial")
        if others:
            last_subjects = [others[0]]

    # numbers: "EPS $1.20 vs $1.35 est" (applies to the companies named in that sentence, or the only one)
    named = sorted((min(a for a, _ in sp), i) for i, sp in target_spans.items() if sp)
    for num in (number_surprise(folded), guidance_surprise(folded)):
        if num is not None and named:  # the numbers belong to the company named first (the one reporting)
            out[named[0][1]].events.insert(0, num)

    for r in out.values():
        _decide(r)
    return out


_DESCRIPTOR = re.compile(r"(?:'s)?[\s-]+(?:partner|supplier|customer|client|rival|peer|competitor|investor|"
                         r"backed|vendor|contractor|affiliate|spinoff|spin-off)\b(?!\s+(?:says|said|with|to|in|on|of)\b)",
                         re.I)
_NEGATED_BEFORE = re.compile(r"\b(?:fails?|failed|did not|didn't|does not|doesn't|unable|not|never|no longer)"
                             r"(?: to)?\s*$", re.I)
# "... in a case brought by Spotify": the company that complained isn't the one in trouble
_COMPLAINANT = re.compile(r"\b(?:brought|filed|lodged|launched) by|\bcomplaint (?:from|by)|\bat the request of|"
                          r"\bafter a complaint (?:from|by)|\bfollowing a complaint (?:from|by)", re.I)
_RATING_BEFORE = re.compile(r"(?:grades?|graded|rates?|rated|rating|initiat\w*|lifts?|raises?|moves?|bumps?|ups|cuts?|"
                            r"lowers?|reiterates?|starts?|resumes?)\b[^.;]{0,50}$", re.I)
_JOIN = re.compile(r"\s*(?:,|and|&|,\s*and|/)\s*", re.I)


def _chained(after: list[int], first_span: dict[int, tuple[int, int]], text: str) -> list[int]:
    """The first company after the verb, plus any joined to it ("upgrades Apple and Microsoft")."""
    if not after:
        return []
    out = [after[0]]
    for prev, nxt in zip(after, after[1:], strict=False):
        if _JOIN.fullmatch(text[first_span[prev][1]:first_span[nxt][0]]):
            out.append(nxt)
        else:
            break
    return out


_NOT_ABBREV = "".join(rf"(?<!\b{a})" for a in (r"U\.S\.", r"U\.K\.", r"E\.U\.", r"Inc\.", r"Corp\.", r"Co\.",
                                                  r"Ltd\.", r"vs\.", r"No\.", r"St\.", r"Mr\.", r"Dr\.", r"adj\.",
                                                  r"Adj\.", r"Jr\.", r"est\.", r"approx\."))


def _clause_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each clause/sentence of the text (headline first)."""
    spans = []
    pos = 0
    for m in re.finditer(r"\s*;\s*|\s+[-–—]\s+|" + _NOT_ABBREV + r"(?<=[.!?])\s+(?=[A-Z0-9\"'$])|\n+|"
                         r",?\s+\b(?:but|while|although|though|"
                         r"however|whereas|despite|even as|yet)\b\s+", text, re.I):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans or [(0, len(text))]


def _decide(r: Reading) -> None:
    """Combine the events for one company into a direction and a strength."""
    not_news = ("law_firm_ad", "recap", "opinion", "roundup")
    if any(f in r.flags for f in not_news):
        r.direction, r.strength = 0, 0.0
        r.neutral_reason = {"law_firm_ad": "a law firm's advert looking for clients, not news",
                            "recap": "describes a move that already happened",
                            "opinion": "an opinion piece or question, not news",
                            "roundup": "a round-up of many stocks"}[next(f for f in not_news if f in r.flags)]
        return
    evs = [e for e in r.events if e.role != "actor"]
    if any(e.kind.startswith("mna_collapse") and e.direction for e in evs):  # "drops bid for X": the bid is over
        evs = [e for e in evs if not e.kind.startswith("mna_target")]
        r.events = [e for e in r.events if not e.kind.startswith("mna_target")]
    if not evs:
        if any(e.role == "actor" for e in r.events):
            r.neutral_reason = "this company is only the one making the move (e.g. the buyer or the analyst firm)"
        elif "moved" in r.flags:
            r.neutral_reason = "only describes a share move - no news event found"
        return
    if any(e.priority == 2 and e.direction for e in evs):  # the forecast is what the market trades, not the quarter
        evs = [e for e in evs if e.priority != 1]
    score = sum(e.direction * e.strength for e in evs)
    directional = [e for e in evs if e.direction]
    if not directional:
        r.neutral_reason = f"{evs[0].label.lower()} - no clear direction for the stock"
        r.events.sort(key=lambda e: -e.priority)
        return
    if abs(score) < 0.15:  # good and bad events cancel out
        r.neutral_reason = "mixed news"
        return
    r.direction = 1 if score > 0 else -1
    same = [e for e in directional if e.direction == r.direction]
    r.strength = min(0.95, max(e.strength for e in same) + 0.04 * (len(same) - 1))
    if "unconfirmed" in r.flags:
        r.strength *= 0.85
    # most important event first (for the explanation)
    r.events.sort(key=lambda e: (-(e.direction == r.direction), -e.priority, -e.strength))


def find_spans(folded: str, symbol: str, terms: list[tuple[str, ...]]) -> list[tuple[int, int]]:
    """Where a company is named in ascii_fold(text): its ticker as written, or any of its name terms."""
    spans = []
    if len(symbol) > 1:
        spans += [m.span() for m in re.finditer(rf"(?<![\w$])\$?{re.escape(symbol)}(?![\w])", folded)]
    else:
        spans += [m.span() for m in re.finditer(rf"\${re.escape(symbol)}(?![\w])", folded)]
    for term in terms:
        rx = r"\b" + r"(?:[\s\-]+|\s*&\s*|\.\s*)".join(re.escape(t) for t in term) + r"(?:'s|s')?\b"
        spans += [m.span() for m in re.finditer(rx, folded, re.I)]
    return sorted(set(spans))


# --------------------------------------------------------------------------------------------- country news
_MACRO_RULES = [
    ("Stimulus / easing", +1, 0.65, r"\b(?:stimulus|easing|rate cuts?|(?:cuts?|lowers?|slashes?) (?:its |the )?"
                                    r"(?:interest |repo |policy |benchmark |key |lending )*rates?|surprise cut|"
                                    r"bond[- ]buying|rescue package|support measures|infrastructure fund|spending "
                                    r"package|loosens? (?:the )?debt brake|fiscal (?:boost|package|expansion))\b"),
    ("Rate hike / tightening", -1, 0.6, r"\b(?:rate hikes?|(?:raises?|hikes?) (?:its |the )?(?:interest |repo |policy |"
                                        r"benchmark |key )*rates?|"
                                        r"tightening|highest (?:rates )?since)\b"),
    ("Economy shrinking", -1, 0.65, r"\b(?:shrinks?|shrank|contracts?|contracted|contraction|recession|slump(?:s|ed)?|"
                                    r"deflation|unexpectedly (?:falls?|drops?|shrinks?)|weakest since)\b"),
    ("Economy growing", +1, 0.55, r"\b(?:grows? faster|beats? (?:growth )?forecasts|stronger[- ]than[- ]expected "
                                  r"(?:growth|gdp)|expands? (?:more|faster) than)\b"),
    ("Political / war risk", -1, 0.7, r"\b(?:martial law|coup|invasion|invades?|war|missile|unrest|impeach\w*|"
                                      r"snap election|political crisis|government collapses?)\b"),
    ("Tariffs / sanctions on it", -1, 0.6, r"\b(?:tariffs? on|sanctions on|export ban on|trade war with)\b"),
    ("Trade deal / ceasefire", +1, 0.6, r"\b(?:trade deal|trade agreement|ceasefire|peace deal|tariff (?:truce|cut|"
                                        r"relief|exemption))\b"),
]
_MACRO_COMPILED = [(lab, d, s, re.compile(p, re.I)) for lab, d, s, p in _MACRO_RULES]


def macro_reading(text: str) -> Reading:
    """International macro news for a country fund (EWJ, FXI...): direction from plain rules, or none."""
    folded = ascii_fold(text)
    r = Reading()
    if _RECAP.search(folded.split("\n", 1)[0]) or _OPINION.search(folded.split("\n", 1)[0]):
        r.flags.append("recap" if _RECAP.search(folded.split("\n", 1)[0]) else "opinion")
        r.neutral_reason = "not fresh news"
        return r
    for label, d, s, rx in _MACRO_COMPILED:
        m = rx.search(folded)
        if m:
            r.events.append(Event("macro", label, d, s, "any", m.group(0), "", "hours"))
    score = sum(e.direction * e.strength for e in r.events)
    if r.events and abs(score) >= 0.15:
        r.direction = 1 if score > 0 else -1
        r.strength = max(e.strength for e in r.events if e.direction == r.direction)
        r.events.sort(key=lambda e: -(e.direction == r.direction))
    elif r.events:
        r.neutral_reason = "mixed news for that country"
    return r
