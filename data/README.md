# Datasets
The datasets consist of string quartet movements by *Joseph Haydn, Wolfgang Amadeus Mozart, and Ludwig van Beethoven*. Each piece is described by the following metadata fields: `filename`, `composer`, `genre`, `opus`, `key`, `movement`, `date`


Genre labels are kept in the form used in the metadata, so labels such as `Minuet`, `Menuetto`, `Scherzando`, and `Tempo di Minuetto` are preserved without additional normalization.

## Small dataset (classic_music_small)


This dataset contains the exact examples listed in Tables 1, 2, and 3 of the original paper by [Mijangos et al. (2022)](https://arxiv.org/pdf/2204.11139).
All MIDI files are stored in a single folder without subdirectories (flat structure).
In this repository, the raw MIDI files are placed in `data/raw/classic_music_small/`, and the corresponding metadata table is stored in `data/metadata/metadata_small.csv`.

Download: [classic_music_small.zip](https://www.dropbox.com/scl/fi/umyirg198bif1xvl1ll7i/classic_music_small_flat.zip?rlkey=4yop73cle8o6uv381jnq8gijh&st=1hjmotzy&dl=0)

| Composer | Genre | Opus | Movement | Key | Date |
|:---|:---|:---|:---:|:---:|---:|
| Beethoven | Scherzo: Allegro molto | Op.18/1 | 3 | F major | 1798 |
| Beethoven | Menuetto: Allegretto | Op.18/4 | 3 | C major | 1798 |
| Beethoven | Scherzo: Allegro | Op.18/6 | 3 | Bb major | 1798 |
| Beethoven | Menuetto: Grazioso | Op.59/3 | 3 | C major | 1805 |
| Beethoven | Scherzando vivace - Presto | Op.127 | 3 | Eb major | 1825 |
| | | | | | |
| Haydn | Menuetto | Op.17/2 | 2 | F major | 1771 |
| Haydn | Menuetto | Op.20/1 | 2 | Eb major | 1772 |
| Haydn | Scherzando | Op.33/3 | 2 | C major | 1781 |
| Haydn | Menuetto | Op.50/3 | 3 | Eb major | 1787 |
| Haydn | Menuetto | Op.64/1 | 2 | C major | 1790 |
| Haydn | Menuetto | Op.71/1 | 3 | Bb major | 1793 |
| Haydn | Menuetto | Op.77/1 | 3 | G major | 1799 |
| | | | | | |
| Mozart | Tempo di Minuetto | No.5 | 3 | F major | 1773 |
| Mozart | Menuetto | No.8 | 3 | F major | 1773 |
| Mozart | Menuetto | No.13 | 3 | D minor | 1773 |
| Mozart | Menuetto | No.14 | 2 | G major | 1782 |
| Mozart | Menuetto | No.19 | 3 | C major | 1785 |
| Mozart | Menuetto-Allegretto | No.23 | 3 | F major | 1790 |



## Large dataset (classic_music_big)


This dataset follows the broader catalogue structure used in the paper by Mijangos et al.(see also the corresponding [list of works](https://github.com/MartinMij/TDA-SQ/blob/main/list%20of%20works.pdf)), but also includes additional movements beyond the focused subset listed above.
The archive is organized by composer (*Haydn, Mozart, Beethoven*), and within each composer by movement genre (`adagio, allegro, minuet, andante, presto, scherzo`).
In this repository, the raw MIDI files are placed in `data/raw/classic_music_big/`, and the corresponding metadata table is stored in `data/metadata/metadata_big.csv`.

Download: [classic_music_big.zip](https://www.dropbox.com/scl/fi/ga2v7ym9yuh2ipt92zu6a/classic_music_genres.zip?rlkey=7n8txpx21lyuof9mhhddnt2ke&st=r0k3l54k&dl=0)

### Beethoven

| Composer  | Genre | Opus | Movement | Key | Date |
|:---|:---|:---|:---:|:---:|---:|
| Beethoven | Adagio | Op.18/1 | 2 | F major | 1798-1800 |
| Beethoven | Adagio | Op.18/2 | 2 | G major | 1798-1800 |
| Beethoven | Adagio | Op.18/6 | 4a | Bb major | 1798-1800 |
| Beethoven | Adagio | Op.59/1 | 3 | F major | 1806 |
| Beethoven | Adagio | Op.59/2 | 2 | E minor | 1806 |
| Beethoven | Adagio | Op.74 | 2 | Eb major | 1809 |
| Beethoven | Adagio | Op.132 | 3 | A minor | 1823-1825 |
| | | | | | |
| Beethoven | Allegro | Op.18/1 | 4 | F major | 1798-1800 |
| Beethoven | Allegro | Op.18/2 | 1 | G major | 1798-1800 |
| Beethoven | Allegro | Op.18/3 | 3 | D major | 1798-1800 |
| Beethoven | Allegro | Op.18/4 | 4 | C minor | 1798-1800 |
| Beethoven | Allegro | Op.18/5 | 1 | A major | 1798-1800 |
| Beethoven | Allegro | Op.59/1 | 1 | F major | 1806 |
| Beethoven | Allegro | Op.59/2 | 1 | E minor | 1806 |
| Beethoven | Allegro | Op.95 | 1 | F minor | 1810-1811 |
| Beethoven | Allegro | Op.130 | 6 | Bb major | 1825 |
| Beethoven | Allegro | Op.131 | 7 | C# minor | 1825-1826 |
| | | | | | |
| Beethoven | Minuet | Op.18/4 | 3 | C minor | 1798-1800 |
| Beethoven | Minuet | Op.18/5 | 2 | A major | 1798-1800 |
| Beethoven | Minuet | Op.59/3 | 3 | C major | 1806 |
| | | | | | |
| Beethoven | Scherzo | Op.18/1 | 3 | F major | 1798-1800 |
| Beethoven | Scherzo | Op.18/2 | 3 | G major | 1798-1800 |
| Beethoven | Scherzo | Op.18/6 | 3 | Bb major | 1798-1800 |
| Beethoven | Scherzo | Op.127 | 3 | Eb major | 1822-1825 |


### Haydn

| Composer  | Genre | Opus | Movement | Key | Date |
|:---|:---|:---|:---:|:---:|---:|
| Haydn | Adagio | Op.01/0 | 3 | Eb major | - |
| Haydn | Adagio | Op.01/1 | 3 | Bb major | - |
| Haydn | Adagio | Op.01/2 | 3 | Eb major | - |
| Haydn | Adagio | Op.01/3 | 1 | D major | - |
| Haydn | Adagio | Op.01/4 | 3 | G major | - |
| Haydn | Adagio | Op.09/2 | 3 | Eb major | 1769 |
| Haydn | Adagio | Op.17/1 | 3 | E major | 1771 |
| Haydn | Adagio | Op.17/2 | 3 | F major | 1771 |
| Haydn | Adagio | Op.17/3 | 3 | Eb major | 1771 |
| Haydn | Adagio | Op.17/5 | 3 | G major | 1771 |
| Haydn | Adagio | Op.20/5 | 3 | F minor | 1772 |
| Haydn | Adagio | Op.20/6 | 2 | A major | 1772 |
| Haydn | Adagio | Op.33/3 | 3 | C major | 1781 |
| Haydn | Adagio | Op.50/2 | 2 | C major | 1787 |
| Haydn | Adagio | Op.54/2 | 4 | C major | 1788 |
| Haydn | Adagio | Op.64/5 | 2 | D major | 1790 |
| Haydn | Adagio | Op.76/4 | 2 | Bb major | 1796-1797 |
| Haydn | Adagio | Op.77/1 | 2 | G major | 1799 |
| | | | | | |
| Haydn | Allegro | Op.01/1 | 1 | Bb major | - |
| Haydn | Allegro | Op.01/2 | 1 | Eb major | - |
| Haydn | Allegro | Op.17/3 | 4 | Eb major | 1771 |
| Haydn | Allegro | Op.20/1 | 1 | Eb major | 1772 |
| Haydn | Allegro | Op.20/3 | 4 | G minor | 1772 |
| Haydn | Allegro | Op.33/1 | 1 | B minor | 1781 |
| Haydn | Allegro | Op.33/3 | 1 | C major | 1781 |
| Haydn | Allegro | Op.50/1 | 1 | Bb major | 1787 |
| Haydn | Allegro | Op.55/2 | 2 | F minor | 1788 |
| Haydn | Allegro | Op.64/4 | 1 | G major | 1790 |
| Haydn | Allegro | Op.64/6 | 1 | Eb major | 1790 |
| Haydn | Allegro | Op.71/1 | 1 | Bb major | 1793 |
| | | | | | |
| Haydn | Minuet | Op.01/0 | 2 | Eb major | - |
| Haydn | Minuet | Op.01/1 | 2 | Bb major | - |
| Haydn | Minuet | Op.01/3 | 4 | D major | - |
| Haydn | Minuet | Op.01/4 | 2 | G major | - |
| Haydn | Minuet | Op.01/6 | 2 | C major | - |
| Haydn | Minuet | Op.09/2 | 2 | Eb major | 1769 |
| Haydn | Minuet | Op.09/3 | 2 | G major | 1769 |
| Haydn | Minuet | Op.17/1 | 2 | E major | 1771 |
| Haydn | Minuet | Op.17/3 | 2 | Eb major | 1771 |
| Haydn | Minuet | Op.17/6 | 2 | D major | 1771 |
| Haydn | Minuet | Op.20/3 | 2 | G minor | 1772 |
| Haydn | Minuet | Op.20/5 | 2 | F minor | 1772 |
| Haydn | Minuet | Op.20/6 | 3 | A major | 1772 |
| Haydn | Minuet | Op.42 | 2 | D minor | 1785 |
| Haydn | Minuet | Op.50/2 | 3 | C major | 1787 |
| Haydn | Minuet | Op.50/5 | 3 | F major | 1787 |
| Haydn | Minuet | Op.54/1 | 3 | G major | 1788 |
| Haydn | Minuet | Op.55/2 | 3 | F minor | 1788 |
| Haydn | Minuet | Op.64/5 | 3 | D major | 1790 |
| Haydn | Minuet | Op.71/1 | 3 | Bb major | 1793 |
| Haydn | Minuet | Op.71/3 | 3 | Eb major | 1793 |
| | | | | | |
| Haydn | Presto | Op.01/0 | 1 | Eb major | - |
| Haydn | Presto | Op.01/0 | 5 | Eb major | - |
| Haydn | Presto | Op.01/1 | 5 | Bb major | - |
| Haydn | Presto | Op.01/2 | 5 | Eb major | - |
| Haydn | Presto | Op.01/3 | 5 | D major | - |
| Haydn | Presto | Op.01/4 | 1 | G major | - |
| Haydn | Presto | Op.01/4 | 5 | G major | - |
| Haydn | Presto | Op.01/6 | 5 | C major | - |
| Haydn | Presto | Op.09/3 | 4 | G major | 1769 |
| Haydn | Presto | Op.17/1 | 1 | E major | 1771 |
| Haydn | Presto | Op.17/1 | 4 | E major | 1771 |
| Haydn | Presto | Op.17/5 | 4 | G major | 1771 |
| Haydn | Presto | Op.20/1 | 4 | Eb major | 1772 |
| Haydn | Presto | Op.33/1 | 4 | B minor | 1781 |
| Haydn | Presto | Op.33/3 | 4 | C major | 1781 |
| Haydn | Presto | Op.33/4 | 4 | Bb major | 1781 |
| Haydn | Presto | Op.42 | 4 | D minor | 1785 |
| Haydn | Presto | Op.54/3 | 4 | E major | 1788 |
| Haydn | Presto | Op.55/2 | 4 | F minor | 1788 |
| Haydn | Presto | Op.55/3 | 4 | Bb major | 1788 |
| Haydn | Presto | Op.64/1 | 4 | C major | 1790 |
| Haydn | Presto | Op.64/2 | 4 | B minor | 1790 |
| Haydn | Presto | Op.64/4 | 4 | G major | 1790 |
| Haydn | Presto | Op.64/6 | 4 | Eb major | 1790 |
| Haydn | Presto | Op.76/5 | 4 | D major | 1796-1797 |
| Haydn | Presto | Op.77/1 | 4 | G major | 1799 |


### Mozart

| Composer  | Genre | Opus | Movement | Key | Date |
|:---|:---|:---|:---:|:---:|---:|
| Mozart | Adagio | No.1 | 1 | G major | 1770 |
| Mozart | Adagio | No.3 | 2 | G major | 1772 |
| Mozart | Adagio | No.7 | 2 | Eb major | 1773 |
| Mozart | Adagio | No.10 | 3 | C major | 1773 |
| Mozart | Adagio | No.12 | 2 | Bb major | 1773 |
| Mozart | Adagio | No.17 | 3 | Bb major | 1784 |
| Mozart | Adagio | No.19 | 1 | C major | 1785 |
| Mozart | Adagio | No.20 | 3 | D major | 1786 |
| | | | | | |
| Mozart | Allegro | No.1 | 2 | G major | 1770 |
| Mozart | Allegro | No.2 | 1 | D major | 1772 |
| Mozart | Allegro | No.4 | 1 | C major | 1772-1773 |
| Mozart | Allegro | No.5 | 1 | F major | 1772-1773 |
| Mozart | Allegro | No.8 | 1 | F major | 1773 |
| Mozart | Allegro | No.8 | 4 | F major | 1773 |
| Mozart | Allegro | No.15 | 1 | D minor | 1783 |
| Mozart | Allegro | No.20 | 4 | D major | 1786 |
| Mozart | Allegro | No.22 | 1 | Bb major | 1790 |
| Mozart | Allegro | No.23 | 4 | F major | 1790 |
| | | | | | |
| Mozart | Andante | No.2 | 2 | D major | 1772 |
| Mozart | Andante | No.4 | 2 | C major | 1772-1773 |
| Mozart | Andante | No.6 | 1 | Bb major | 1773 |
| Mozart | Andante | No.8 | 2 | F major | 1773 |
| Mozart | Andante | No.11 | 3 | Eb major | 1773 |
| Mozart | Andante | No.15 | 2 | D minor | 1783 |
| Mozart | Andante | No.18 | 3 | A major | 1785 |
| Mozart | Andante | No.21 | 2 | D major | 1789 |
| Mozart | Andante | No.23 | 2 | F major | 1790 |
| | | | | | |
| Mozart | Minuet | No.1 | 3 | G major | 1770 |
| Mozart | Minuet | No.5 | 3 | F major | 1772-1773 |
| Mozart | Minuet | No.8 | 3 | F major | 1773 |
| Mozart | Minuet | No.9 | 3 | A major | 1773 |
| Mozart | Minuet | No.10 | 2 | C major | 1773 |
| Mozart | Minuet | No.11 | 2 | Eb major | 1773 |
| Mozart | Minuet | No.13 | 3 | D minor | 1773 |
| Mozart | Minuet | No.14 | 2 | G major | 1782 |
| Mozart | Minuet | No.16 | 3 | Eb major | 1783 |
| Mozart | Minuet | No.18 | 2 | A major | 1785 |
| Mozart | Minuet | No.19 | 3 | C major | 1785 |
| Mozart | Minuet | No.23 | 3 | F major | 1790 |
