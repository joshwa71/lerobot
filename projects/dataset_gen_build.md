# libero_ext build report

Built by `scripts/libero_tasks/build_suite.py` from `scripts/libero_tasks/plan.json` (planner seed 4). Checks per task: stable settled resets (drift < 1 cm), goal false at start, forced goal reached without interpenetration, objects in the agentview image, end effector within 5 cm of each approach point, and the scripted expert succeeding in at least 2 of 3 episodes. 50 start states per task.

**60 tasks in 27 of 27 scenes built.**

## S01 (kitchen): built after 14 layout(s), 299 s

Classes: bbq sauce = bbq_sauce, bowl = akita_black_bowl, cabinet = wooden_cabinet, cookies = cookies, fridge = short_fridge, mug = white_yellow_mug.

- put the cookies in the bowl: drift 0.00 cm; reach (cm) above cookies 2.7, bowl 2.8; 50 resets for the start states; expert 3/3
- put the bbq sauce in the fridge and put the mug on the cabinet: drift 0.00 cm; reach (cm) above bbq sauce 2.2, fridge opening 2.8, above mug 2.5, cabinet top 2.8; 50 resets for the start states; expert 3/3

Rejected layouts (last few): #8 put the bbq sauce in the fridge and put the mug on the cabinet: {'drift': 0.0, 'low': -0.009950777323967874, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #9 put the bbq sauce in the fridge and put the mug on the cabinet: {'drift': 0.0, 'low': -0.009950777323967874, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #10 put the bbq sauce in the fridge and put the mug on the cabinet: {'drift': 1.0, 'low': -0.004327470178351223, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #11 put the bbq sauce in the fridge and put the mug on the cabinet: {'drift': 1.0, 'low': -0.902394171198558, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #12 put the bbq sauce in the fridge and put the mug on the cabinet: expert 0/3

## S02 (kitchen): built after 1 layout(s), 59 s

Classes: butter = butter, fridge = short_fridge, plate = plate, salad dressing = salad_dressing.

- put the butter on the plate and put the salad dressing in the fridge: drift 0.00 cm; reach (cm) above butter 2.7, plate 2.4, above salad dressing 2.3, fridge opening 2.9; 50 resets for the start states; expert 3/3

## S03 (kitchen): built after 2 layout(s), 126 s

Classes: bbq sauce = bbq_sauce, cabinet = white_cabinet, cream cheese = cream_cheese, ketchup = ketchup, plate = plate, wine bottle = wine_bottle.

- put the bbq sauce in the bottom drawer and put the ketchup in front of the plate: drift 0.00 cm; reach (cm) above bbq sauce 2.4, drawer 2.6, above ketchup 2.8, beside 2.6; 50 resets for the start states; expert 3/3
- put the cream cheese on the plate and put the wine bottle on the cabinet: drift 0.00 cm; reach (cm) above cream cheese 2.3, plate 2.3, above wine bottle 2.3, cabinet top 2.3; 50 resets for the start states; expert 3/3

Rejected layouts (last few): #0 layout: no non-overlapping layout

## S04 (kitchen): built after 2 layout(s), 176 s

Classes: alphabet soup = alphabet_soup, cabinet = wooden_cabinet, drainer = bowl_drainer, faucet = faucet, macaroni = macaroni_and_cheese, ramekin = glazed_rim_porcelain_ramekin.

- put the macaroni on the cabinet: drift 0.00 cm; reach (cm) above macaroni 2.6, cabinet top 2.9; 50 resets for the start states; expert 3/3
- turn on the faucet: drift 0.00 cm; reach (cm) faucet 2.1; 50 resets for the start states; expert 2/3
- put the alphabet soup in the top drawer and put the ramekin in the right side of the drainer: drift 0.00 cm; reach (cm) above alphabet soup 3.0, drawer 2.4, above ramekin 2.4, drainer 2.2; 50 resets for the start states; expert 3/3

Rejected layouts (last few): #0 put the macaroni on the cabinet: {'drift': 1.6289053439422219e-15, 'low': -0.004237494667114539, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S05 (kitchen): built after 17 layout(s), 418 s

Classes: bowl = akita_black_bowl, drainer = bowl_drainer, fridge = short_fridge, popcorn = popcorn, tomato sauce = tomato_sauce.

- put the popcorn in the bowl: drift 0.00 cm; reach (cm) above popcorn 2.6, bowl 2.3; 50 resets for the start states; expert 3/3
- put the tomato sauce in the fridge and put the bowl in the left side of the drainer: drift 0.00 cm; reach (cm) above tomato sauce 2.7, fridge opening 2.5, above bowl 2.8, drainer 2.6; 50 resets for the start states; expert 2/3

Rejected layouts (last few): #11 put the tomato sauce in the fridge and put the bowl in the left side of the drainer: expert 0/3; #12 put the tomato sauce in the fridge and put the bowl in the left side of the drainer: expert 0/3; #13 put the popcorn in the bowl: {'drift': 2.2487262639618315e-15, 'low': -0.902394171198558, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #14 put the tomato sauce in the fridge and put the bowl in the left side of the drainer: expert 0/3; #15 put the tomato sauce in the fridge and put the bowl in the left side of the drainer: expert 1/3

## S06 (kitchen): built after 3 layout(s), 61 s

Classes: book = black_book, cabinet = wooden_cabinet, moka pot = moka_pot, tray = wooden_tray.

- put the book on the cabinet and put the moka pot in the tray: drift 0.00 cm; reach (cm) above book 2.2, cabinet top 2.8, above moka pot 2.6, tray 2.7; 50 resets for the start states; expert 2/3

Rejected layouts (last few): #0 put the book on the cabinet and put the moka pot in the tray: {'drift': 0.0, 'low': -0.002586208943078261, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #1 put the book on the cabinet and put the moka pot in the tray: {'drift': np.float64(1.0408340855860843e-16), 'low': -0.00258620894307815, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S07 (kitchen): built after 11 layout(s), 310 s

Classes: cabinet = wooden_cabinet, macaroni = macaroni_and_cheese, microwave = microwave, placemat = dining_set_group, ramekin = glazed_rim_porcelain_ramekin, salad dressing = salad_dressing, tomato sauce = tomato_sauce.

- put the tomato sauce in the middle drawer: drift 0.00 cm; reach (cm) above tomato sauce 2.7, drawer 2.6; 51 resets for the start states; expert 3/3
- put the macaroni in the microwave and close it: drift 0.00 cm; reach (cm) above macaroni 2.6, microwave opening 2.4; 52 resets for the start states; expert 3/3
- put the salad dressing on the cabinet and put the ramekin on the placemat: drift 0.00 cm; reach (cm) above salad dressing 2.6, cabinet top 2.8, above ramekin 2.7, placemat 2.7; 51 resets for the start states; expert 3/3

Rejected layouts (last few): #5 put the salad dressing on the cabinet and put the ramekin on the placemat: {'drift': 1.6896206656014101e-15, 'low': -0.004667130382987494, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #6 layout: no non-overlapping layout; #7 put the salad dressing on the cabinet and put the ramekin on the placemat: {'drift': 1.7496550317114955e-15, 'low': -0.004667130382987383, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #8 layout: no non-overlapping layout; #9 put the salad dressing on the cabinet and put the ramekin on the placemat: {'drift': 1.0547118733938987e-15, 'low': -0.004667130382987494, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S08 (kitchen): built after 5 layout(s), 287 s

Classes: bowl = akita_black_bowl, butter = butter, cabinet = wooden_cabinet, microwave = microwave, orange juice = orange_juice, ramekin = glazed_rim_porcelain_ramekin.

- put the orange juice to the right of the bowl: drift 0.00 cm; reach (cm) above orange juice 2.7, beside 2.4; 50 resets for the start states; expert 3/3
- put the ramekin in the bowl: drift 0.00 cm; reach (cm) above ramekin 3.0, bowl 2.8; 50 resets for the start states; expert 3/3
- put the bowl in the microwave and put the butter on the cabinet: drift 0.00 cm; reach (cm) above bowl 3.0, microwave opening 2.4, above butter 3.0, cabinet top 2.9; 50 resets for the start states; expert 2/3

Rejected layouts (last few): #0 put the bowl in the microwave and put the butter on the cabinet: {'drift': 0.0, 'low': -0.0028077126930423635, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #1 put the bowl in the microwave and put the butter on the cabinet: {'drift': 7.152448122690996e-15, 'low': -0.0028077126930423635, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #2 put the bowl in the microwave and put the butter on the cabinet: expert 0/3; #3 put the bowl in the microwave and put the butter on the cabinet: {'drift': 0.0, 'low': -0.0028077126930423635, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S09 (kitchen): built after 9 layout(s), 278 s

Classes: alphabet soup = alphabet_soup, cream cheese = cream_cheese, fridge = short_fridge, frying pan = chefmate_8_frypan, ketchup = ketchup, macaroni = macaroni_and_cheese, stove = flat_stove, tray = wooden_tray.

- lift the alphabet soup: drift 0.00 cm; reach (cm) above alphabet soup 2.6; 50 resets for the start states; expert 3/3
- put the ketchup in the fridge and put the macaroni in the tray: drift 0.00 cm; reach (cm) above ketchup 2.7, fridge opening 2.9, above macaroni 2.3, tray 2.6; 50 resets for the start states; expert 3/3
- turn on the stove and put the cream cheese in the pan: drift 0.00 cm; reach (cm) above cream cheese 2.6, pan 3.0; 50 resets for the start states; expert 3/3

Rejected layouts (last few): #3 layout: no non-overlapping layout; #4 turn on the stove and put the cream cheese in the pan: {'drift': 1.6514567491299204e-15, 'low': -0.004237494667114539, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #5 put the ketchup in the fridge and put the macaroni in the tray: expert 0/3; #6 layout: no non-overlapping layout; #7 lift the alphabet soup: {'drift': 1.0, 'low': -0.9094105427057643, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}

## S10 (study): built after 12 layout(s), 191 s

Classes: book = yellow_book, cabinet = white_cabinet, milk = milk, orange juice = orange_juice, shelf = wooden_two_layer_shelf.

- put the milk on the cabinet: drift 0.00 cm; reach (cm) above milk 2.6, cabinet top 2.5; 50 resets for the start states; expert 3/3
- put the book in the bottom drawer and put the orange juice on the shelf: drift 0.00 cm; reach (cm) above book 2.8, drawer 2.8, above orange juice 2.8, shelf 2.6; 50 resets for the start states; expert 3/3

Rejected layouts (last few): #6 put the milk on the cabinet: {'drift': 3.5493527563892463e-15, 'low': -0.0006632570903359891, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #7 put the book in the bottom drawer and put the orange juice on the shelf: expert 1/3; #8 put the milk on the cabinet: {'drift': 3.469446951953614e-15, 'low': -0.0006632570903359891, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #9 put the book in the bottom drawer and put the orange juice on the shelf: {'drift': 0.0, 'low': -0.0006632570903359891, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #10 put the milk on the cabinet: {'drift': 6.938893903907228e-15, 'low': -0.0006632570903359891, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S11 (study): built after 6 layout(s), 217 s

Classes: cabinet = white_cabinet, cookies = cookies, ketchup = ketchup, mug = white_yellow_mug, orange juice = orange_juice, shelf = wooden_two_layer_shelf, storage box = white_storage_box.

- put the cookies in the middle drawer: drift 0.00 cm; reach (cm) above cookies 2.3, drawer 2.7; 50 resets for the start states; expert 2/3
- put the mug on the shelf: drift 0.00 cm; reach (cm) above mug 2.4, shelf 2.8; 50 resets for the start states; expert 3/3
- put the ketchup on the storage box and put the orange juice on the cabinet: drift 0.00 cm; reach (cm) above ketchup 2.4, storage box 2.8, above orange juice 2.3, cabinet top 2.2; 50 resets for the start states; expert 2/3

Rejected layouts (last few): #0 put the cookies in the middle drawer: {'drift': 3.469446951953614e-15, 'low': -0.002195654584990603, 'start_success': False, 'forced': False, 'view': True, 'reach': {'above cookies': 0.077, 'drawer': 0.028}}; #1 put the mug on the shelf: {'drift': 3.0269169999247415e-15, 'low': -0.002195654584990603, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #2 put the cookies in the middle drawer: {'drift': 2.063081476116683e-16, 'low': -0.002195654584990603, 'start_success': False, 'forced': False, 'view': True, 'reach': {'above cookies': 0.051, 'drawer': 0.022}}; #3 put the ketchup on the storage box and put the orange juice on the cabinet: {'drift': 3.469446951953614e-15, 'low': -0.002195654584990603, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #4 put the cookies in the middle drawer: {'drift': 1.0, 'low': -0.003568543703407645, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}

## S12 (study): built after 23 layout(s), 255 s

Classes: alphabet soup = alphabet_soup, butter = butter, cabinet = wooden_cabinet, caddy = desk_caddy, moka pot = moka_pot, shelf = wooden_two_layer_shelf.

- put the alphabet soup on the cabinet: drift 0.00 cm; reach (cm) above alphabet soup 2.3, cabinet top 2.3; 50 resets for the start states; expert 3/3
- put the moka pot on the shelf and put the butter in the left of the caddy: drift 0.00 cm; reach (cm) above moka pot 2.9, shelf 2.8, above butter 2.4, caddy 2.3; 50 resets for the start states; expert 3/3

Rejected layouts (last few): #17 put the alphabet soup on the cabinet: {'drift': 5.412337245047638e-16, 'low': -0.002064979391585786, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #18 put the moka pot on the shelf and put the butter in the left of the caddy: expert 1/3; #19 put the alphabet soup on the cabinet: {'drift': 1.304512053934559e-15, 'low': -0.002064979391585786, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #20 put the moka pot on the shelf and put the butter in the left of the caddy: {'drift': 2.8895890415671624e-15, 'low': -0.002064979391585897, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #21 put the alphabet soup on the cabinet: {'drift': 1.8735013540549517e-15, 'low': -0.002064979391585897, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S13 (study): built after 3 layout(s), 244 s

Classes: book = yellow_book, bowl = white_bowl, butter = butter, plate = plate, ramekin = glazed_rim_porcelain_ramekin, storage box = white_storage_box, tomato sauce = tomato_sauce.

- lift the bowl: drift 0.00 cm; reach (cm) above bowl 2.7; 50 resets for the start states; expert 3/3
- put the book behind the bowl and put the ramekin on the plate: drift 0.00 cm; reach (cm) above book 2.7, beside 2.8, above ramekin 2.4, plate 3.0; 50 resets for the start states; expert 3/3
- put the butter in the bowl and put the tomato sauce on the storage box: drift 0.00 cm; reach (cm) above butter 2.5, bowl 2.9, above tomato sauce 2.2, storage box 2.7; 50 resets for the start states; expert 2/3

Rejected layouts (last few): #0 lift the bowl: expert 0/3; #1 lift the bowl: expert 1/3

## S14 (study): built after 3 layout(s), 152 s

Classes: book = black_book, caddy = desk_caddy, cream cheese = cream_cheese, ketchup = ketchup, shelf = wooden_two_layer_shelf, storage box = white_storage_box.

- put the book on the storage box: drift 0.00 cm; reach (cm) above book 2.9, storage box 2.2; 50 resets for the start states; expert 2/3
- put the ketchup on the shelf and put the cream cheese in the right of the caddy: drift 0.00 cm; reach (cm) above ketchup 2.5, shelf 2.6, above cream cheese 3.0, caddy 2.4; 50 resets for the start states; expert 3/3

Rejected layouts (last few): #0 put the book on the storage box: expert 1/3; #1 put the ketchup on the shelf and put the cream cheese in the right of the caddy: {'drift': 1.1102230246251565e-16, 'low': -0.00044176484168612085, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S15 (living room): built after 3 layout(s), 100 s

Classes: bbq sauce = bbq_sauce, ramekin = glazed_rim_porcelain_ramekin, shelf = wooden_two_layer_shelf, tray = wooden_tray.

- put the bbq sauce on the lower shelf: drift 0.00 cm; reach (cm) above bbq sauce 2.9, shelf opening 2.7; 50 resets for the start states; expert 2/3
- put the ramekin in the tray: drift 0.00 cm; reach (cm) above ramekin 2.6, tray 2.9; 50 resets for the start states; expert 3/3

Rejected layouts (last few): #0 put the bbq sauce on the lower shelf: {'drift': 1.1102230246251565e-13, 'low': -0.43651730399149, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #1 put the ramekin in the tray: {'drift': 0.0, 'low': 0.00025714010990196545, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S16 (living room): built after 1 layout(s), 236 s

Classes: basket = basket, chocolate pudding = chocolate_pudding, cookies = cookies, milk = milk, mug = red_coffee_mug, placemat = dining_set_group, plate = plate, storage box = white_storage_box.

- put the cookies on the plate: drift 0.00 cm; reach (cm) above cookies 2.8, plate 2.5; 50 resets for the start states; expert 3/3
- put the milk on the storage box: drift 0.00 cm; reach (cm) above milk 2.5, storage box 2.5; 50 resets for the start states; expert 3/3
- put the mug on the placemat and put the chocolate pudding in the basket: drift 0.00 cm; reach (cm) above mug 2.4, placemat 2.5, above chocolate pudding 2.8, basket 2.9; 50 resets for the start states; expert 3/3

## S17 (living room): built after 1 layout(s), 105 s

Classes: bbq sauce = bbq_sauce, bookcase = wooden_shelf, cream cheese = cream_cheese, popcorn = popcorn, storage box = white_storage_box, tray = wooden_tray.

- put the cream cheese on the storage box: drift 0.00 cm; reach (cm) above cream cheese 2.7, storage box 2.5; 50 resets for the start states; expert 3/3
- put the bbq sauce in the tray and put the popcorn in the bookcase: drift 0.00 cm; reach (cm) above bbq sauce 2.3, tray 2.5, above popcorn 2.4, bookcase opening 2.6; 50 resets for the start states; expert 3/3

## S18 (coffee table): built after 1 layout(s), 122 s

Classes: cookies = cookies, milk = milk, shelf = wooden_two_layer_shelf, storage box = white_storage_box, wine bottle = wine_bottle.

- lift the milk: drift 0.00 cm; reach (cm) above milk 2.8; 50 resets for the start states; expert 3/3
- put the wine bottle on the shelf and put the cookies on the storage box: drift 0.00 cm; reach (cm) above wine bottle 2.8, shelf 2.4, above cookies 2.4, storage box 2.8; 50 resets for the start states; expert 3/3

## S19 (coffee table): built after 1 layout(s), 279 s

Classes: bowl = white_bowl, chocolate pudding = chocolate_pudding, milk = milk, moka pot = moka_pot, popcorn = popcorn, ramekin = glazed_rim_porcelain_ramekin, shelf = wooden_two_layer_shelf, storage box = white_storage_box, tray = wooden_tray.

- lift the moka pot: drift 0.00 cm; reach (cm) above moka pot 2.4; 50 resets for the start states; expert 2/3
- put the chocolate pudding in the bowl and put the milk in the tray: drift 0.00 cm; reach (cm) above chocolate pudding 2.6, bowl 2.4, above milk 3.0, tray 2.2; 50 resets for the start states; expert 3/3
- put the ramekin on the storage box and put the popcorn on the shelf: drift 0.00 cm; reach (cm) above ramekin 2.2, storage box 2.9, above popcorn 2.6, shelf 3.0; 50 resets for the start states; expert 3/3

## S20 (coffee table): built after 1 layout(s), 136 s

Classes: macaroni = macaroni_and_cheese, plate = plate, popcorn = popcorn, shelf = wooden_two_layer_shelf.

- put the macaroni on the shelf: drift 0.00 cm; reach (cm) above macaroni 2.7, shelf 3.0; 50 resets for the start states; expert 3/3
- put the popcorn on the plate: drift 0.00 cm; reach (cm) above popcorn 2.2, plate 2.5; 50 resets for the start states; expert 3/3

## S21 (tabletop): built after 1 layout(s), 84 s

Classes: chocolate pudding = chocolate_pudding, cookies = cookies, storage box = white_storage_box, tray = wooden_tray.

- put the chocolate pudding on the storage box: drift 0.00 cm; reach (cm) above chocolate pudding 2.5, storage box 2.9; 50 resets for the start states; expert 3/3
- put the cookies in the tray: drift 0.00 cm; reach (cm) above cookies 3.0, tray 2.6; 50 resets for the start states; expert 3/3

## S22 (tabletop): built after 2 layout(s), 193 s

Classes: basket = basket, bbq sauce = bbq_sauce, bowl = akita_black_bowl, butter = butter, cream cheese = cream_cheese, frying pan = chefmate_8_frypan.

- put the butter in the pan: drift 0.00 cm; reach (cm) above butter 2.5, pan 2.9; 50 resets for the start states; expert 3/3
- put the bbq sauce in the basket and put the cream cheese in the bowl: drift 0.00 cm; reach (cm) above bbq sauce 2.3, basket 2.3, above cream cheese 2.4, bowl 2.6; 50 resets for the start states; expert 2/3

Rejected layouts (last few): #0 put the bbq sauce in the basket and put the cream cheese in the bowl: expert 1/3

## S23 (tabletop): built after 39 layout(s), 361 s

Classes: basket = basket, chocolate pudding = chocolate_pudding, macaroni = macaroni_and_cheese, moka pot = moka_pot, orange juice = orange_juice, plate = plate, shelf = wooden_two_layer_shelf, tomato sauce = tomato_sauce, tray = wooden_tray.

- put the orange juice in the tray: drift 0.00 cm; reach (cm) above orange juice 2.8, tray 2.2; 50 resets for the start states; expert 2/3
- put the macaroni in the basket and put the moka pot to the right of the plate: drift 0.00 cm; reach (cm) above macaroni 2.2, basket 2.2, above moka pot 2.6, beside 2.9; 51 resets for the start states; expert 3/3
- put the tomato sauce on the shelf and put the chocolate pudding on the plate: drift 0.00 cm; reach (cm) above tomato sauce 3.0, shelf 2.6, above chocolate pudding 2.4, plate 2.6; 50 resets for the start states; expert 3/3

Rejected layouts (last few): #33 layout: no non-overlapping layout; #34 put the tomato sauce on the shelf and put the chocolate pudding on the plate: expert 0/3; #35 layout: no non-overlapping layout; #36 layout: no non-overlapping layout; #37 layout: no non-overlapping layout

## S24 (tabletop): built after 2 layout(s), 149 s

Classes: basket = basket, caddy = desk_caddy, chocolate pudding = chocolate_pudding, popcorn = popcorn, wine bottle = wine_bottle.

- put the popcorn in the basket: drift 0.00 cm; reach (cm) above popcorn 2.6, basket 2.7; 50 resets for the start states; expert 3/3
- put the wine bottle to the left of the basket and put the chocolate pudding in the left of the caddy: drift 0.00 cm; reach (cm) above wine bottle 2.4, beside 2.9, above chocolate pudding 3.0, caddy 2.8; 50 resets for the start states; expert 3/3

Rejected layouts (last few): #0 put the wine bottle to the left of the basket and put the chocolate pudding in the left of the caddy: {'drift': 0.0, 'low': -0.012126986942515239, 'start_success': False, 'forced': False, 'view': True, 'reach': {'above wine bottle': 0.028, 'beside': 0.022, 'above chocolate pudding': 0.073, 'caddy': 0.024}}

## S25 (floor): built after 2 layout(s), 169 s

Classes: alphabet soup = alphabet_soup, basket = basket, cookies = cookies, mug = red_coffee_mug, storage box = white_storage_box, tray = wooden_tray.

- put the cookies in the basket: drift 0.00 cm; reach (cm) above cookies 2.5, basket 2.8; 50 resets for the start states; expert 3/3
- put the mug in the tray and put the alphabet soup on the storage box: drift 0.00 cm; reach (cm) above mug 2.9, tray 2.6, above alphabet soup 2.3, storage box 2.5; 50 resets for the start states; expert 3/3

Rejected layouts (last few): #0 put the mug in the tray and put the alphabet soup on the storage box: expert 1/3

## S26 (kitchen): built after 7 layout(s), 115 s

Classes: microwave = microwave, mug = white_yellow_mug, plate = plate, popcorn = popcorn.

- put the mug to the left of the plate and put the popcorn in the microwave: drift 0.00 cm; reach (cm) above mug 2.4, beside 2.2, above popcorn 2.9, microwave opening 2.4; 50 resets for the start states; expert 2/3

Rejected layouts (last few): #1 put the mug to the left of the plate and put the popcorn in the microwave: expert 0/3; #2 layout: no non-overlapping layout; #3 put the mug to the left of the plate and put the popcorn in the microwave: {'drift': 1.0, 'low': -0.004327470178351223, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #4 put the mug to the left of the plate and put the popcorn in the microwave: expert 0/3; #5 layout: no non-overlapping layout

## S27 (kitchen): built after 3 layout(s), 132 s

Classes: cabinet = white_cabinet, fridge = short_fridge, popcorn = popcorn, wine bottle = wine_bottle.

- put the wine bottle in the fridge: drift 0.00 cm; reach (cm) above wine bottle 2.3, fridge opening 2.7; 50 resets for the start states; expert 3/3
- open the top drawer and put the popcorn in it: drift 0.00 cm; reach (cm) above popcorn 2.9, drawer front 2.9; 50 resets for the start states; expert 2/3

Rejected layouts (last few): #0 put the wine bottle in the fridge: expert 1/3; #1 open the top drawer and put the popcorn in it: expert 1/3
