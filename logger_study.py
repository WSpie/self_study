import logging

logger = logging.getLogger('SATOSHI')
# This config si not recommended when several scripts needed, and only the first defined one counts!!!
# logging.basicConfig(filename='poke.log', level=logging.DEBUG, format='%(asctime)s:%(name)s:%(message)s')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('poke.log')
logger.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

class Pokemon:
    def __init__(self, id=0, name=None, type1=None, type2=None):
        self.id = id
        self.name = name
        self.type1 = type1
        self.type2 = type2
        logger.debug(f'Pokemon added: {self.id}-{self.name}-{self.type}')
    @property
    def type(self):
        return (f'{self.type1}+{self.type2}' if self.type2 else f'{self.type1}')
    
    def get_id(self):
        try:
            res = self.id / 0
        except:
            logger.exception('Error')
        else:
            return res

poke1 = Pokemon(25, 'Pikachu', 'Electric')
poke2 = Pokemon(6, 'Charizard', 'Fire', 'Flying')
poke3 = Pokemon()
print(poke1.get_id(), poke2.get_id())
