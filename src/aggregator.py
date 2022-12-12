from logger import Logger

import os
import typing
import re

logger = Logger()

def aggregate_raw_transcripts(
        data_directory: str,
        raw_transcript_dir_name: str,
        dataset_name: str,
        Filter_data: bool = False,
        dataset_extension: str = ".txt",
        force: bool = False
    ) -> None:
    """
    Parse raw transcript files downloaded from LINK.
    Then aggregate into a one full, intermediate dataset text file.
    """
    
    transcript_dataset_filepath = f"{os.path.join(data_directory, dataset_name)}{dataset_extension}"
    
    # exit early if corpus exists and we don't want to force re-create the file
    if os.path.isfile(transcript_dataset_filepath) and not force:
        logger.log_info(f"dataset '{transcript_dataset_filepath}' already exists, using cached version instead. (use force=True to overwrite existing dataset)")
        return
    
    logger.log_info(f"aggregating individual transcripts to full corpus, stored in '{transcript_dataset_filepath}'")
    raw_data_path = os.path.join(data_directory, raw_transcript_dir_name, dataset_name)
    with open(transcript_dataset_filepath, "w") as dataset_f:
        for file in os.listdir(raw_data_path):
            filepath = os.path.join(raw_data_path, file)
            
            # read raw transcripts and write to aggregated dataset file
            with open(filepath, "r") as f:
                data = [line.strip() for line in f.readlines()]
                
                if Filter_data:
                    censored_dict = {
                        'as*ault,' : 'assaoult,', 'assh*le.' : 'asshole', 'w*r' : 'war', 'h*tler,' : 'hitler,', 
                        'assh*le!' :'asshole!', 'h*m*.' : 'homo.', 'f*ck!' : "fuck!", 'b*at,' : 'brat,', 
                        'assh*le,' : 'asshole,', 'a*tillery' : 'artillery', '"assh*le.”' : '"asshole."', 't*nk' : 'tank', 
                        't*nk.' : 'tank.', 'm*rder' : 'murder', 'k*ll?' : 'kill?', 'k*ll,' : 'kill,', 
                        'w*r.' : 'war.', 't*nk?' : 'tank?', 'g*n?' : 'gun?', 'sh*t.' : 'shot.', 
                        'assh*le."' : 'asshole."', 'f*ck.' : 'fuck.', 'w*apon!' : 'weapon!', '"f*ck' : '"fuck', 
                        'assh*le' : 'asshole', 'p*rn,' : 'porn,', 'm*rd' : 'murd', 'b*ll*ts?' : 'bullets?', 
                        'as*ault' : 'assoult', 'g*n.' : 'gun.', 'k*ll.' : 'kill.', 'b*mb?' : 'bomb?', 
                        'g*n*t' : 'gunshot', 'p*stol' : 'pistol', 'g*n' : 'gun', 'b*at?' : 'brat?', 
                        'a**l' : 'ass', 'b*mb' : 'bomb', 'f*g' : 'fag', 'a*t*matic' : 'automatic', 
                        'g*n,' : 'gun,', 'b*at' : 'brat', 'sh*thole?' : 'shithole?', 'expl*sive.' : 'explosive', 
                        'm*rder?' : 'murder', 'g*dd*mn.' : 'goddamn.', 'b*mb,' : 'bomb,', 'f*ck' : 'fuck', 
                        'n*zi,' : 'nazi,', 'expl*si*n' : 'explosion', 't*nk,' : 'tank,', 'w*apon.' : 'weapon.', 
                        'sh**ting' : 'shooting', 'b*ll*ts.' : 'bullets', 'expl*si*n.' : 'explosive.', 'p*stol,' : 'pistol,', 
                        'k*ll' : 'kill', 'assh*le?' : 'asshole?', "h*tler's" : "hitler's", 'w*apon?' : 'weapon?', 
                        'g*n!' : 'gun!', 'b*llet' : 'bullet', 'r*ped' : 'raped', 'sh**ting.' : 'shootting.', 
                        'sh**t' : 'shoot', 'b*at.' : 'brat.', 'sh*t,' : 'shot,', "t*nk's" : "tank's", 
                        "b*at'" : "brat'", 'sh*t' : 'shit', 'expl*sives,' : 'explosives', 'b*rned' : 'burned', 
                        'w*apon,' : 'weapon,', 'sh**t.' : 'shoot.', 'f*cking' : 'fucking', 'b*mb.' : 'bomb.',
                        'g*dd*mn' : 'goddamn', 'w*apon' : 'weapon', 'm*rder.' : 'murder.', 'p*ssy.' : 'pussy.',
                        'expl*si*n?' : 'explosion', 'b*ll*ts' : 'bullets', 'sh**t?' : 'shoot?', 'r*fle.' : 'rifle.',
                        'g*ngb*ng' : 'gangbang', 'f*cked' : 'fucked', 'p*ssy' : 'pussy', 'n*zi' : 'nazi',
                        'm*rder,' : 'murder,', 'sn*per' : 'sniper', '*' : '*', 'h*tler' : 'hitler', 'sh*t?' : 'shit?',
                        'b*llet.' : 'bullet.'}
                    data = filter_dialog(data, censored_dict)
                
                for stripped_line in data:
                    if len(stripped_line) != 0:
                        dataset_f.write(f"{stripped_line}\n")
    logger.log_info("finished aggregating transcripts")
                        
                        
                        
def filter_dialog(lines: typing.List[str], cd: typing.Dict) -> typing.List[str]:
    """
    Remove '[speaker]:' tokens from dialogue.
    Lines is a list of strings, where each string should represent one continuous group of lines spoken by a character.
    """
    regex_pattern = r"([^:]+:\s*)"
    filler = r"((\(|\[)([^()]+)(\)|\]))"
    scene_line = r"((SCENE|scene|Scene):.*)"
    special_edge_case = r"\*throws up\*"
    unwanted = r"^(Submitted and corrected by:)|^(Last season on AMC's Breaking Bad)|^(Previously,* on AMC's Breaking Bad)"
    stripped_lines: typing.List[str] = []
    for line in lines:
        # append words that do not match the speaker token
        if (re.match(scene_line,line) is None) and (re.match(unwanted,line) is None):
            new_line = re.sub(filler, '', line )
            new_line = re.sub(regex_pattern, '', new_line )
            new_line = re.sub(special_edge_case, '', new_line)
            spaces = re.findall("\s{1,}", new_line)
            words = re.split("\s{1,}", new_line)
            updated_words = words.copy()
            c_flag = False
            for i, word in enumerate(words):
                if '*' in word:
                    c_flag = True
                    updated_words[i] = cd[word]
            if c_flag:
                out = [x+y for x,y in zip(updated_words, spaces)]
                final_string = ''
                for word in out:
                    final_string += word
                stripped_lines.append(final_string)
            else:
                stripped_lines.append(new_line)
    return stripped_lines