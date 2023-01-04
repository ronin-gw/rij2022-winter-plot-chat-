#!/usr/bin/env python3
import json
import re
import os.path
import pickle
import argparse
from datetime import datetime, timezone, timedelta
from collections import Counter
from itertools import chain
from multiprocessing import Pool
from operator import itemgetter
from copy import copy

from sudachipy import tokenizer, dictionary
import jaconv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties

from adjustText import adjust_text

from emoji import EMOJI_DATA

matplotlib.use("module://mplcairo.macosx")

TIMELINE = os.path.join(os.path.dirname(__file__), "timeline.pickle")
TIMEZONE = timezone(timedelta(hours=9), "JST")

matplotlib.rcParams["font.sans-serif"] = ["Hiragino Maru Gothic Pro", "Yu Gothic", "Meirio", "Takao", "IPAexGothic", "IPAPGothic", "VL PGothic", "Noto Sans CJK JP"]
emoji_prop = FontProperties(fname="/System/Library/Fonts/Apple Color Emoji.ttc")

UNICODE_EMOJI = EMOJI_DATA.keys()

# (ward to plot, line style, color)
RTA_EMOTES = (
    ("rtaClap", "-", "#ec7087"),
    ("rtaPray", "-", "#f7f97a"),
    (("rtaGl", "GL"), "-", "#5cc200"),
    (("rtaGg", "GG"), "-", "#ff381c"),
    ("rtaCheer", "-", "#ffbe00"),
    ("rtaHatena", "-", "#ffb5a1"),
    ("rtaR", "-", "white"),
    (("rtaCry", "BibleThump"), "-", "#5ec6ff"),

    ("rtaListen", "-.", "#5eb0ff"),
    ("rtaKabe", "-.", "#bf927a"),
    ("rtaFear", "-.", "#8aa0ec"),
    (("rtaRedbull", "rtaRedbull2", "レッドブル"), "-.", "#98b0df"),
    ("rtaPokan", "-.", "#838187"),
    ("rtaGogo", "-.", "#df4f69"),
    # ("rtaBanana", ":", "#f3f905"),
    # ("rtaBatsu", ":", "#5aafdd"),
    # ("rtaShogi", ":", "#c68d46"),
    ("rtaThink", ":", "#c68d46"),
    # ("rtaIizo", ":", "#0f9619"),

    ("rtaHello", "-.", "#ff3291"),
    ("rtaHmm", "-.", "#fcc7b9"),
    ("rtaPog", "-.", "#f8c900"),
    ("rtaMaru", ":", "#c80730"),
    ("rtaFire", ":", "#E56124"),
    ("rtaIce", ":", "#CAEEFA"),
    # ("rtaThunder", ":", "#F5D219"),
    # ("rtaPoison", ":", "#9F65B2"),
    ("rtaGlitch", ":", "#9F65B2"),
    # ("rtaWind", ":", "#C4F897"),
    # ("rtaOko", "-.", "#d20025"),
    ("rtaWut", ":", "#d97f8d"),
    ("rtaPolice", ":", "#7891b8"),
    # ("rtaChan", "-.", "green"),
    # ("rtaKappa", "-.", "#ffeae2"),

    # ("rtaSleep", "-.", "#ff8000"),
    # ("rtaCafe", "--", "#a44242"),
    # ("rtaDot", "--", "#ff3291"),

    # ("rtaShi", ":", "#8aa0ec"),
    # ("rtaGift", ":", "white"),
    # ("rtaAnkimo", ":", "#f92218 "),

    # ("rtaFrameperfect", "--", "#ff7401"),
    # ("rtaPixelperfect", "--", "#ffa300"),
    (("草", "ｗｗｗ", "LUL"), "--", "#1e9100"),
    ("無敵時間", "--", "red"),
    ("Cheer", "--", "#bd62fe"),
    ("石油王", "--", "yellow"),
    ("かわいい", "--", "#ff3291"),

)
VOCABULARY = set(w for w, _, _, in RTA_EMOTES if isinstance(w, str))
VOCABULARY |= set(chain(*(w for w, _, _, in RTA_EMOTES if isinstance(w, tuple))))

EXCLUDE_MESSAGE_TERMS = (
    " subscribed with Prime",
    " subscribed at Tier ",
    " gifted a Tier ",
    " is gifting "
)

# (title, movie start time as timestamp, offset hour, min, sec)
GAMES = (
    ("\n\n始まりのあいさつ", 1671990862.384, 0, 4, 44),
    ("バーガーバーガー", 1671990862.384, 0, 8, 1),
    ("アンリミテッド:サガ", 1671990862.384, 1, 9, 13),
    ("デュープリズム", 1671990862.384, 3, 4, 29),
    ("スターオーシャン4 -THE LAST HOPE- 4K & Full HD Remaster", 1671990862.384, 6, 2, 9, "right"),
    ("バトルネットワーク ロックマンエグゼ", 1671990862.384, 8, 48, 53),
    ("オーディンスフィアレイヴスラシル", 1671990862.384, 10, 36, 43),
    ("BRAVELY\nDEFAULT II", 1671990862.384, 12, 24, 8),
    ("星のカービィ Wii", 1671990862.384, 15, 52, 39),
    ("ヨッシーウールワールド", 1671990862.384, 18, 39, 53),
    ("New スーパーマリオブラザーズ 2", 1671990862.384, 21, 27, 33),
    ("進め！キノピオ隊長", 1671990862.384, 22, 35, 53),
    ("スーパーマリオパーティ", 1671990862.384, 24, 32, 37),
    ("箱庭えくすぷろーらもあ", 1671990862.384, 26, 21, 4),
    ("ZADETTE", 1671990862.384, 27, 47, 21),
    ("heavenly bodies", 1671990862.384, 28, 25, 0),
    ("Stardew Valley", 1671990862.384, 29, 12, 2),
    ("クッキークリッカー", 1671990862.384, 31, 59, 44),
    ("とっとこハム太郎3 ラブラブ大冒険でちゅ", 1671990862.384, 33, 30, 0),
    ("海腹川背・旬", 1671990862.384, 35, 12, 2, "right"),
    ("Untitled\nGoose Game\n〜いたずらガチョウがやって来た！〜", 1671990862.384, 35, 35, 28, "right"),
    ("高橋名人の大冒険島\n(Super Adventure Island)", 1671990862.384, 35, 58, 32),
    ("Mighty Gunvolt Burst", 1671990862.384, 36, 29, 23),
    ("METAL GEAR SOLID V: THE PHANTOM PAIN", 1671990862.384, 37, 27, 12),
    ("ポケットモンスター エメラルド", 1672136715.093, 0, 6, 18),
    ("サンリオタイムネット未来編", 1672136715.093, 2, 49, 8),
    ("ファイアーエムブレム 封印の剣", 1672136715.093, 5, 32, 35),
    ("ダージュ オブ ケルベロス\n-ファイナルファンタジーVII- インターナショナル", 1672136715.093, 7, 26, 59),
    ("スーパーマリオ ３Dワールド ＋ フューリーワールド", 1672136715.093, 9, 10, 41),
    ("メタルスラッグ4", 1672136715.093, 10, 56, 51),
    ("ALTF4", 1672136715.093, 11, 28, 40),
    ("Pogostuck:\nRage With Your Friends", 1672136715.093, 11, 53, 1),
    ("不思議な夢の海のとばり", 1672136715.093, 12, 30, 13),
    ("Cuphead", 1672136715.093, 14, 0, 30),
    ("ソニック3 & ナックルズ", 1672136715.093, 15, 31, 32),
    ("A Dance of Fire and Ice", 1672136715.093, 16, 43, 19),
    ("BIOHAZARD5", 1672136715.093, 17, 52, 15),
    ("METAL GEAR 2: SOLID SNAKE(MSX復刻版)", 1672136715.093, 20, 3, 48),
    ("大工の源さん\n～べらんめ町騒動記～", 1672136715.093, 21, 30, 28),
    ("不思議のダンジョン 風来のシレン外伝 女剣士アスカ見参! for Windows", 1672136715.093, 22, 3, 21),
    ("Metal Unit", 1672136715.093, 25, 48, 25),
    ("鉄騎", 1672136715.093, 26, 41, 23),
    ("カスタムロボ バトルレボリューション", 1672136715.093, 27, 33, 24),
    ("ドラゴンクエストモンスターズ\nテリーのワンダーランドRETRO", 1672136715.093, 29, 58, 21, "right"),
    ("ドラゴンクエストソード\n仮面の女王と鏡の塔", 1672136715.093, 31, 23, 31, "right"),
    ("風のクロノア1 アンコール", 1672136715.093, 33, 54, 43),
    ("ポコニャン！\nへんぽこりんアドベンチャー", 1672136715.093, 34, 55, 56),
    ("スターフォックスアサルト", 1672136715.093, 35, 32, 40),
    ("Metal Slug XX", 1672136715.093, 36, 30, 20),
    ("ノットトレジャーハンター", 1672136715.093, 37, 6, 35),
    ("そろそろ寿司を食べないと\n死ぬぜ！", 1672136715.093, 38, 27, 1),
    ("サイレントヒルシリーズ UFOリレー", 1672136715.093, 39, 20, 8),
    ("The Convenience Store | 夜勤事件", 1672136715.093, 41, 56, 12, "right"),
    ("悪魔城ドラキュラX\n血の輪廻", 1672136715.093, 42, 30, 29, "right"),
    ("Ato", 1672136715.093, 42, 59, 50),
    ("COGEN:\n大鳥こはくと刻の剣", 1672136715.093, 43, 33, 37),
    ("Factorio", 1672295014.258, 0, 9, 58),
    ("Dead Rising", 1672295014.258, 1, 56, 34),
    ("ゼルダの伝説 ブレスオブザワイルド", 1672295014.258, 2, 44, 52),
    ("ピクミン3 デラックス", 1672295014.258, 6, 8, 32),
    ("真・三國無双7 with 猛将伝", 1672295014.258, 7, 26, 33),
    ("Salt and Sacrifice", 1672295014.258, 8, 38, 7),
    ("Demon’s Souls (2020)", 1672295014.258, 10, 1, 36),
    ("カドゥケウス NEW BLOOD", 1672295014.258, 11, 2, 59),
    ("弾幕アマノジャク\n～ Impossible Spell Card.", 1672295014.258, 13, 8, 45, "right"),
    ("迷宮組曲\nミロンの大冒険", 1672295014.258, 14, 39, 3),
    ("Batman :\nThe Video Game", 1672295014.258, 15, 15, 25),
    ("パズルボブル", 1672295014.258, 15, 44, 49),
    ("Petal Crash", 1672295014.258, 16, 20, 57),
    ("キャサリン", 1672295014.258, 16, 59, 10),
    ("だるま道場", 1672295014.258, 17, 58, 49),
    ("ロックマン10\n宇宙からの脅威!!", 1672295014.258, 19, 3, 43, "right"),
    ("ロックマンゼクス", 1672295014.258, 20, 0, 10, "right"),
    ("Inscryption", 1672295014.258, 21, 14, 13),
    ("零~刺青ノ聲~ Fatal Frame Ⅲ The Tormented", 1672295014.258, 22, 36, 22),
    ("FINAL FANTASY X HD Remaster", 1672295014.258, 25, 28, 2),
    ("トランシルビィ", 1672295014.258, 34, 41, 55),
    ("ポケモンスナップ", 1672295014.258, 36, 18, 11),
    ("MOTHER2 ギーグの逆襲", 1672295014.258, 36, 57, 36),
    ("スーパーマリオカート", 1672295014.258, 39, 32, 11, "right"),
    ("F-ZERO", 1672295014.258, 40, 27, 46),
    ("Metroid Fusion", 1672295014.258, 41, 59, 3),
    ("スーパードンキーコングシリーズ トリロジーリレー", 1672295014.258, 44, 0, 31),
    ("終わりのあいさつ", 1672295014.258, 46, 11, 14, "right")
)


class Game:
    def __init__(self, name, t, h, m, s, align="left"):
        self.name = name
        self.startat = datetime.fromtimestamp(t + h * 3600 + m * 60 + s)
        self.align = align


GAMES = tuple(Game(*args) for args in GAMES)

WINDOWSIZE = 1
WINDOW = timedelta(seconds=WINDOWSIZE)
AVR_WINDOW = 60
PER_SECONDS = 60
FIND_WINDOW = 15
DOMINATION_RATE = 0.6
COUNT_THRESHOLD = 37.5

DPI = 200
ROW = 5
PAGES = 4
YMAX = 700
WIDTH = 3840
HEIGHT = 2160


FONT_COLOR = "white"
FRAME_COLOR = "#ffff79"
BACKGROUND_COLOR = "#352319"
FACE_COLOR = "#482b1e"
ARROW_COLOR = "#ffff79"
MESSAGE_FILL_COLOR = "#1e0d0b"
MESSAGE_EDGE_COLOR = "#7f502f"

BACKGROUND = "rijw.png"


class Message:
    _tokenizer = dictionary.Dictionary().create()
    _mode = tokenizer.Tokenizer.SplitMode.C

    pns = (
        "無敵時間",
        "石油王",
        "躊躇しないでください",
        "国境なき医師団",
        "ナイスセーブ",
        "ナイスセーヌ",
        "クッキークリッカー",
        "バーガーバーガー",
        "ガーバーガーバー",
        "ハイプトレイン",
        "ちーぷる",
        "死んだぜ",
        "心折設計",
        "飛鳥文化",
        "キマリは通さない",
        "インド人を右に",
        "無双時間",
        "せきゆくん",
        "設計ミス",
        "恋にドロップドロップ",
        "目的なんですか"
    )
    pn_patterns = (
        (re.compile("[\u30A1-\u30FF]+ケンカ"), "〜ケンカ"),
        (re.compile("[a-zA-Z]+[0-9]+"), "Cheer"),
        (re.compile("世界[1１一]位?"), "世界一")
    )
    stop_words = (
        "Squid2",
        "する",
        ''
    )

    @classmethod
    def _tokenize(cls, text):
        return cls._tokenizer.tokenize(text, cls._mode)

    def __init__(self, raw):
        # self.name = raw["author"]["name"]

        if "emotes" in raw:
            self.emotes = set(e["name"] for e in raw["emotes"]
                              if e["name"] not in self.stop_words)
        else:
            self.emotes = set()
        self.datetime = datetime.fromtimestamp(int(raw["timestamp"]) // 1000000).replace(tzinfo=TIMEZONE)

        self.message = raw["message"]
        self.msg = set()

        message = self.message
        for emote in self.emotes:
            message = message.replace(emote, "")
        for stop in self.stop_words:
            message = message.replace(stop, "")

        #
        for pattern, replace in self.pn_patterns:
            match = pattern.findall(message)
            if match:
                self.msg.add(replace)
                if pattern.pattern.startswith('^') and pattern.pattern.endswith('$'):
                    message = ''
                else:
                    for m in match:
                        message = message.replace(m, "")

        #
        for pn in self.pns:
            if pn in message:
                self.msg.add(pn)
                message = message.replace(pn, "")

        #
        message = jaconv.h2z(message)

        # (名詞 or 動詞) (+助動詞)を取り出す
        parts = []
        currentpart = None
        for m in self._tokenize(message):
            part = m.part_of_speech()[0]

            if currentpart:
                if part == "助動詞":
                    parts.append(m.surface())
                else:
                    self.msg.add(''.join(parts))
                    parts = []
                    if part in ("名詞", "動詞"):
                        currentpart = part
                        parts.append(m.surface())
                    else:
                        currentpart = None
            else:
                if part in ("名詞", "動詞"):
                    currentpart = part
                    parts.append(m.surface())

        if parts:
            self.msg.add(''.join(parts))

        #
        kusa = False
        for word in copy(self.msg):
            if set(word) & set(('w', 'ｗ')):
                kusa = True
                self.msg.remove(word)
        if kusa:
            self.msg.add("ｗｗｗ")

        message = message.strip()
        if not self.msg and message:
            self.msg.add(message)

    def __len__(self):
        return len(self.msg)

    @property
    def words(self):
        return self.msg | self.emotes


def _make_messages(raw_message):
    for term in EXCLUDE_MESSAGE_TERMS:
        if term in raw_message["message"]:
            return
    return Message(raw_message)


def _parse_chat(paths):
    messages = []
    for p in paths:
        with open(p) as f, Pool() as pool:
            j = json.load(f)
            messages += [msg for msg in pool.map(_make_messages, j, len(j) // pool._processes)
                         if msg is not None]

    timeline = []
    currentwindow = messages[0].datetime.replace(microsecond=0) + WINDOW
    _messages = []
    for m in messages:
        if m.datetime <= currentwindow:
            _messages.append(m)
        else:
            timeline.append((currentwindow, *_make_timepoint(_messages)))
            while True:
                currentwindow += WINDOW
                if m.datetime <= currentwindow:
                    _messages = [m]
                    break
                else:
                    timeline.append((currentwindow, 0, Counter()))

    if _messages:
        timeline.append((currentwindow, *_make_timepoint(_messages)))

    return timeline


def _make_timepoint(messages):
    total = len(messages)
    counts = Counter(_ for _ in chain(*(m.words for m in messages)))

    return total, counts


def _load_timeline(paths):
    if os.path.exists(TIMELINE):
        with open(TIMELINE, "rb") as f:
            timeline = pickle.load(f)
    else:
        timeline = _parse_chat(paths)
        with open(TIMELINE, "wb") as f:
            pickle.dump(timeline, f)

    return timeline


def _save_counts(timeline):
    _, _, counters = zip(*timeline)

    counter = Counter()
    for c in counters:
        counter.update(c)

    with open("words.tab", 'w') as f:
        for w, c in sorted(counter.items(), key=itemgetter(1), reverse=True):
            print(w, c, sep='\t', file=f)


def _plot(timeline):
    for npage in range(1, 1 + PAGES):
        chunklen = int(len(timeline) / PAGES / ROW)

        fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR
        ax = fig.add_axes((0, 0, 1, 1))
        background_image = mpimg.imread(BACKGROUND)
        ax.imshow(background_image)

        plt.subplots_adjust(left=0.07, bottom=0.05, top=0.92)

        for i in range(1, 1 + ROW):
            nrow = i + ROW * (npage - 1)
            f, t = chunklen * (nrow - 1), chunklen * nrow
            x, c, y = zip(*timeline[f:t])
            _x = tuple(t.replace(tzinfo=None) for t in x)

            ax = fig.add_subplot(ROW, 1, i)
            _plot_row(ax, _x, y, c, i == 1, i == ROW)

        fig.suptitle(f"RTA in Japan Winter 2022 チャット頻出スタンプ・単語 ({npage}/{PAGES})",
                     color=FONT_COLOR, size="x-large")
        fig.text(0.03, 0.5, "単語 / 分 （同一メッセージ内の重複は除外）",
                 ha="center", va="center", rotation="vertical", color=FONT_COLOR, size="large")
        fig.savefig(f"{npage}.png", dpi=DPI, transparent=True)
        plt.close()
        print(npage)


def moving_average(x, w=AVR_WINDOW):
    _x = np.convolve(x, np.ones(w), "same") / w
    return _x[:len(x)]


def _plot_row(ax, x, y, total_raw, add_upper_legend, add_lower_legend):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M", tz=TIMEZONE))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(range(0, 60, 5)))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.set_facecolor(FACE_COLOR)
    for axis in ("top", "bottom", "left", "right"):
        ax.spines[axis].set_color(FRAME_COLOR)

    ax.tick_params(colors=FONT_COLOR, which="both")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, YMAX)
    # ax.set_ylim(25, 800)
    # ax.set_yscale('log')

    total = moving_average(total_raw) * PER_SECONDS
    total = ax.fill_between(x, 0, total, color=MESSAGE_FILL_COLOR,
                            edgecolor=MESSAGE_EDGE_COLOR, linewidth=0.3)

    for i, game in enumerate(GAMES):
        if x[0] <= game.startat <= x[-1]:
            ax.axvline(x=game.startat, color=ARROW_COLOR, linestyle=":")
            ax.annotate(game.name, xy=(game.startat, YMAX), xytext=(game.startat, YMAX * 0.85), verticalalignment="top",
                        color=FONT_COLOR, arrowprops=dict(facecolor=ARROW_COLOR, shrink=0.05), ha=game.align)

    # ys = []
    # labels = []
    # colors = []
    for words, style, color in RTA_EMOTES:
        if isinstance(words, str):
            words = (words, )
        _y = np.fromiter((sum(c[w] for w in words) for c in y), int)
        if not sum(_y):
            continue
        _y = moving_average(_y) * PER_SECONDS
        # ys.append(_y)
        # labels.append("\n".join(words))
        # colors.append(color if color else None)
        ax.plot(x, _y, label="\n".join(words), linestyle=style, color=(color if color else None))
    # ax.stackplot(x, ys, labels=labels, colors=colors)

    #
    avr_10min = moving_average(total_raw, FIND_WINDOW) * FIND_WINDOW
    words = Counter()
    for counter in y:
        words.update(counter)
    words = set(k for k, v in words.items() if v >= COUNT_THRESHOLD)
    words -= VOCABULARY

    annotations = []
    for word in words:
        at = []
        _ys = moving_average(np.fromiter((c[word] for c in y), int), FIND_WINDOW) * FIND_WINDOW
        for i, (_y, total_y) in enumerate(zip(_ys, avr_10min)):
            if _y >= total_y * DOMINATION_RATE and _y >= COUNT_THRESHOLD:
                at.append((i, _y * PER_SECONDS / FIND_WINDOW))
        if at:
            at.sort(key=lambda x: x[1])
            at = at[-1]

            if any(c in UNICODE_EMOJI for c in word):
                text = ax.text(x[at[0]], at[1], word, color=FONT_COLOR, fontsize="xx-small", fontproperties=emoji_prop)
            else:
                text = ax.text(x[at[0]], at[1], word, color=FONT_COLOR, fontsize="xx-small")
            annotations.append(text)
    adjust_text(annotations, only_move={"text": 'x'})

    if add_upper_legend:
        leg = ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0)
        _set_legend(leg)

    if add_lower_legend:
        leg = plt.legend([total], ["メッセージ / 分"], loc=(1.015, 0.4), framealpha=0)
        _set_legend(leg)
        msg = "図中の単語は{}秒間で{}%の\nメッセージに含まれていた単語\n({:.1f}メッセージ / 秒 以上のもの)".format(
            FIND_WINDOW, int(DOMINATION_RATE * 100), COUNT_THRESHOLD / FIND_WINDOW
        )
        plt.gcf().text(0.915, 0.06, msg, fontsize="x-small", color=FONT_COLOR)


def _set_legend(leg):
    frame = leg.get_frame()
    # frame.set_facecolor(FACE_COLOR)
    frame.set_edgecolor(FRAME_COLOR)

    for text in leg.get_texts():
        text.set_color(FONT_COLOR)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+")
    args = parser.parse_args()

    timeline = _load_timeline(args.json)
    _save_counts(timeline)
    _plot(timeline)


if __name__ == "__main__":
    _main()
