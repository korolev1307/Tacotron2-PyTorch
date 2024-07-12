""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict

# _pad        = '_'
# _punctuation = '!\'(),.:;? '
# _special = '-'
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ['@' + s for s in cmudict.valid_symbols]

# # Export all symbols:
# symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

_pad        = '▁'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = ['a','b','bʲ','d','dʲ','dʲː','dː','d͡z','d͡zʲ','d͡ʐ','e','f','fʲ','i','j','jː',
                    'k','kʲ','kː','lʲ','lʲː','m','mʲ','mʲː','mː','n','nʲ','nʲː','nː','o','p','pʲ','pʲː','pː','r',
                    'rʲ','rʲː','rː','s','sʲ','sʲː','sː','t','tʲ','tʲː','tː','t͡s','t͡sʲ','t͡sː','t͡ɕ','t͡ɕː','t͡ʂ',
                    'u','uˑ','v','vʲ','vʲː','vː','x','xʲ','z','zʲ','zː','æ','ɐ','ɑː','ɕ','ɕː','ə','ɛ','ɛ̠','ɡ','ɡʲ',
                    'ɡː','ɣ','ɨ','ɪ','ɫ','ɫː','ɵ','ɾ','ʂ','ʂː','ʉ','ʊ','ʐ','ʐː','ʑː','ʔ','⁽ʲ ⁾','⁽ʲ ⁾ː']
symbols = [_pad] + list(_special) + list(_punctuation) + _letters