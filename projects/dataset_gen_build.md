# libero_ext build report

Built by `scripts/libero_tasks/build_suite.py` from `scripts/libero_tasks/plan.json` (planner seed 41). Checks per task: stable settled resets (drift < 1 cm), goal false at start, forced goal reached without interpenetration, objects in the agentview image, end effector within 5 cm of each approach point. 50 start states per task.

**60 tasks in 25 of 25 scenes built.**

## S01 (kitchen): built after 1 layout(s), 125 s

Classes: butter = butter, cookies = cookies, faucet = faucet, fridge = short_fridge, tray = wooden_tray.

- turn on the faucet: drift 0.00 cm; reach (cm) faucet 2.6; 50 resets for the start states
- put the butter in the fridge and put the cookies in the tray: drift 0.00 cm; reach (cm) above butter 2.7, fridge opening 2.7, above cookies 2.6, tray 2.7; 50 resets for the start states

## S02 (kitchen): built after 19 layout(s), 251 s

Classes: bbq sauce = bbq_sauce, bowl = white_bowl, drainer = bowl_drainer, fridge = short_fridge, mug = porcelain_mug, placemat = dining_set_group, ramekin = glazed_rim_porcelain_ramekin, salad dressing = new_salad_dressing.

- put the ramekin in the bowl: drift 0.00 cm; reach (cm) above ramekin 2.6, bowl 2.9; 50 resets for the start states
- put the bowl in the left side of the drainer and put the mug on the placemat: drift 0.00 cm; reach (cm) above bowl 3.0, drainer 2.3, above mug 2.7, placemat 2.8; 50 resets for the start states
- put the salad dressing in the fridge and put the bbq sauce in front of the bowl: drift 0.00 cm; reach (cm) above salad dressing 2.4, fridge opening 2.4, above bbq sauce 2.9, beside 2.9; 50 resets for the start states

Rejected layouts (last few): #13 put the bowl in the left side of the drainer and put the mug on the placemat: {'drift': 0.0, 'low': -0.032603141691782134, 'start_success': False, 'forced': False, 'view': True, 'reach': {'above bowl': 0.027, 'drainer': 0.023, 'above mug': 0.024, 'placemat': 0.041}}; #14 layout: no non-overlapping layout; #15 layout: no non-overlapping layout; #16 layout: no non-overlapping layout; #17 layout: no non-overlapping layout

## S03 (kitchen): built after 1 layout(s), 69 s

Classes: cabinet = white_cabinet, orange juice = orange_juice, tomato sauce = tomato_sauce, tray = wooden_tray.

- put the tomato sauce in the bottom drawer and put the orange juice in the tray: drift 0.00 cm; reach (cm) above tomato sauce 2.5, drawer 3.0, above orange juice 2.3, tray 2.3; 50 resets for the start states

## S04 (kitchen): built after 4 layout(s), 255 s

Classes: bowl = akita_black_bowl, cabinet = wooden_cabinet, chocolate pudding = chocolate_pudding, drainer = bowl_drainer, milk = milk, ramekin = glazed_rim_porcelain_ramekin, tray = wooden_tray, wine bottle = wine_bottle.

- put the chocolate pudding in the bowl: drift 0.00 cm; reach (cm) above chocolate pudding 2.9, bowl 2.3; 50 resets for the start states
- put the ramekin in the right side of the drainer: drift 0.00 cm; reach (cm) above ramekin 2.5, drainer 2.4; 50 resets for the start states
- put the milk in the tray and put the wine bottle on the cabinet: drift 0.00 cm; reach (cm) above milk 2.5, tray 2.4, above wine bottle 2.5, cabinet top 2.7; 50 resets for the start states

Rejected layouts (last few): #0 put the chocolate pudding in the bowl: {'drift': 2.8189256484623115e-15, 'low': -0.0026072886421317776, 'start_success': False, 'forced': False, 'view': True, 'reach': {'above chocolate pudding': 0.052, 'bowl': 0.025}}; #1 put the chocolate pudding in the bowl: {'drift': 0.0, 'low': -0.0026072886421317776, 'start_success': False, 'forced': False, 'view': True, 'reach': {'above chocolate pudding': 0.153, 'bowl': 0.024}}; #2 put the chocolate pudding in the bowl: {'drift': 1.0, 'low': -0.5824568679706068, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}

## S05 (kitchen): built after 2 layout(s), 94 s

Classes: chocolate pudding = chocolate_pudding, fridge = short_fridge, orange juice = orange_juice, plate = plate, tomato sauce = tomato_sauce.

- put the orange juice to the right of the plate: drift 0.00 cm; reach (cm) above orange juice 2.7, beside 3.0; 50 resets for the start states
- put the tomato sauce in the fridge and put the chocolate pudding on the plate: drift 0.00 cm; reach (cm) above tomato sauce 2.3, fridge opening 2.9, above chocolate pudding 2.9, plate 2.3; 50 resets for the start states

Rejected layouts (last few): #0 layout: no non-overlapping layout

## S06 (kitchen): built after 9 layout(s), 232 s

Classes: bbq sauce = bbq_sauce, cabinet = wooden_cabinet, fridge = short_fridge, macaroni = macaroni_and_cheese, microwave = microwave, salad dressing = salad_dressing, wine bottle = wine_bottle.

- put the bbq sauce in the top drawer: drift 0.00 cm; reach (cm) above bbq sauce 2.4, drawer 2.7; 50 resets for the start states
- put the wine bottle in the fridge: drift 0.00 cm; reach (cm) above wine bottle 2.8, fridge opening 2.9; 50 resets for the start states
- put the salad dressing on the cabinet and put the macaroni in the microwave: drift 0.00 cm; reach (cm) above salad dressing 2.5, cabinet top 2.6, above macaroni 2.7, microwave opening 2.7; 50 resets for the start states

Rejected layouts (last few): #3 put the salad dressing on the cabinet and put the macaroni in the microwave: {'drift': 1.0, 'low': -0.002517303991490416, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #4 layout: no non-overlapping layout; #5 layout: no non-overlapping layout; #6 layout: no non-overlapping layout; #7 put the salad dressing on the cabinet and put the macaroni in the microwave: {'drift': 6.938893903907228e-15, 'low': -0.002517303991490416, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S07 (kitchen): built after 18 layout(s), 181 s

Classes: cookies = cookies, fridge = short_fridge, ketchup = ketchup, microwave = microwave, mug = white_yellow_mug, plate = plate, popcorn = popcorn, tray = wooden_tray.

- put the cookies on the plate: drift 0.00 cm; reach (cm) above cookies 2.9, plate 2.4; 50 resets for the start states
- open the fridge and put the ketchup in it: drift 0.00 cm; reach (cm) above ketchup 2.3, fridge opening 4.3; 50 resets for the start states
- put the popcorn in the microwave and put the mug in the tray: drift 0.00 cm; reach (cm) above popcorn 2.9, microwave opening 2.9, above mug 2.7, tray 2.6; 50 resets for the start states

Rejected layouts (last few): #12 layout: no non-overlapping layout; #13 layout: no non-overlapping layout; #14 layout: no non-overlapping layout; #15 layout: no non-overlapping layout; #16 layout: no non-overlapping layout

## S08 (study): built after 1 layout(s), 203 s

Classes: caddy = desk_caddy, cookies = cookies, mug = white_yellow_mug, ramekin = glazed_rim_porcelain_ramekin, salad dressing = salad_dressing, shelf = wooden_two_layer_shelf, storage box = white_storage_box.

- lift the salad dressing: drift 0.00 cm; reach (cm) above salad dressing 2.6; 50 resets for the start states
- put the mug on the shelf: drift 0.00 cm; reach (cm) above mug 2.4, shelf 2.8; 50 resets for the start states
- put the cookies on the storage box and put the ramekin in the left of the caddy: drift 0.00 cm; reach (cm) above cookies 3.0, storage box 2.7, above ramekin 2.9, caddy 2.6; 50 resets for the start states

## S09 (study): built after 1 layout(s), 181 s

Classes: bookcase = wooden_shelf, butter = butter, caddy = desk_caddy, cream cheese = cream_cheese, ketchup = ketchup, shelf = wooden_two_layer_shelf.

- put the butter in the right of the caddy: drift 0.00 cm; reach (cm) above butter 2.7, caddy 2.6; 50 resets for the start states
- put the cream cheese in the bookcase: drift 0.00 cm; reach (cm) above cream cheese 2.5, bookcase opening 2.8; 50 resets for the start states
- put the ketchup on the shelf: drift 0.00 cm; reach (cm) above ketchup 2.5, shelf 2.8; 50 resets for the start states

## S10 (study): built after 1 layout(s), 135 s

Classes: bookcase = wooden_shelf, bowl = akita_black_bowl, cream cheese = cream_cheese, popcorn = popcorn.

- put the cream cheese in the bowl: drift 0.00 cm; reach (cm) above cream cheese 2.3, bowl 2.7; 50 resets for the start states
- put the popcorn in the bookcase: drift 0.00 cm; reach (cm) above popcorn 2.4, bookcase opening 3.0; 50 resets for the start states

## S11 (study): built after 1 layout(s), 211 s

Classes: bbq sauce = bbq_sauce, bookcase = wooden_shelf, cabinet = white_cabinet, caddy = desk_caddy, chocolate pudding = chocolate_pudding, cream cheese = cream_cheese, moka pot = moka_pot.

- lift the moka pot: drift 0.00 cm; reach (cm) above moka pot 2.9; 50 resets for the start states
- put the chocolate pudding in the bookcase: drift 0.00 cm; reach (cm) above chocolate pudding 2.9, bookcase opening 2.4; 50 resets for the start states
- put the bbq sauce on the cabinet and put the cream cheese in the right of the caddy: drift 0.00 cm; reach (cm) above bbq sauce 2.6, cabinet top 2.8, above cream cheese 2.8, caddy 2.9; 50 resets for the start states

## S12 (study): built after 17 layout(s), 266 s

Classes: alphabet soup = alphabet_soup, book = yellow_book, cabinet = white_cabinet, caddy = desk_caddy, chocolate pudding = chocolate_pudding, moka pot = moka_pot, plate = plate, popcorn = popcorn, storage box = white_storage_box.

- put the chocolate pudding in the left of the caddy: drift 0.00 cm; reach (cm) above chocolate pudding 2.7, caddy 2.2; 52 resets for the start states
- put the alphabet soup on the storage box and put the moka pot in front of the plate: drift 0.00 cm; reach (cm) above alphabet soup 2.7, storage box 2.9, above moka pot 2.9, beside 2.3; 56 resets for the start states
- put the popcorn on the plate and put the book on the cabinet: drift 0.00 cm; reach (cm) above popcorn 2.6, plate 2.5, above book 2.2, cabinet top 3.0; 53 resets for the start states

Rejected layouts (last few): #11 layout: no non-overlapping layout; #12 layout: no non-overlapping layout; #13 layout: no non-overlapping layout; #14 layout: no non-overlapping layout; #15 put the chocolate pudding in the left of the caddy: {'drift': 4.819661020294401, 'low': -0.8997872052073743, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}

## S13 (study): built after 10 layout(s), 201 s

Classes: book = yellow_book, cabinet = white_cabinet, ketchup = ketchup, milk = milk, mug = red_coffee_mug, shelf = wooden_two_layer_shelf, storage box = white_storage_box.

- put the book in the bottom drawer and put the milk on the shelf: drift 0.00 cm; reach (cm) above book 2.4, drawer 2.6, above milk 2.4, shelf 2.5; 51 resets for the start states
- put the ketchup on the storage box and put the mug on the cabinet: drift 0.00 cm; reach (cm) above ketchup 2.8, storage box 2.3, above mug 2.4, cabinet top 3.0; 56 resets for the start states

Rejected layouts (last few): #4 put the book in the bottom drawer and put the milk on the shelf: {'drift': 6.994405055138486e-15, 'low': -0.007815279234344552, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #5 put the book in the bottom drawer and put the milk on the shelf: {'drift': 1.0, 'low': -0.007815279234344552, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #6 put the book in the bottom drawer and put the milk on the shelf: {'drift': 1.0, 'low': -0.007815279234344663, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #7 put the ketchup on the storage box and put the mug on the cabinet: {'drift': 3.469446951953614e-15, 'low': -0.007815279234344552, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #8 put the ketchup on the storage box and put the mug on the cabinet: {'drift': 6.938893903907228e-15, 'low': -0.007815279234344552, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S14 (living room): built after 4 layout(s), 150 s

Classes: cream cheese = cream_cheese, frying pan = chefmate_8_frypan, moka pot = moka_pot, tray = wooden_tray.

- put the cream cheese in the pan: drift 0.00 cm; reach (cm) above cream cheese 2.5, pan 2.7; 54 resets for the start states
- put the moka pot in the tray: drift 0.00 cm; reach (cm) above moka pot 2.4, tray 2.9; 51 resets for the start states

Rejected layouts (last few): #0 put the cream cheese in the pan: {'drift': 1.0, 'low': -0.459536637500857, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #1 put the cream cheese in the pan: {'drift': 1.0, 'low': -0.46375903954850284, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #2 put the cream cheese in the pan: {'drift': 1.0, 'low': -0.02699291612654431, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}

## S15 (living room): built after 20 layout(s), 306 s

Classes: basket = basket, bbq sauce = bbq_sauce, book = black_book, chocolate pudding = chocolate_pudding, milk = milk, storage box = white_storage_box, tray = wooden_tray.

- lift the milk: drift 0.00 cm; reach (cm) above milk 2.5; 50 resets for the start states
- put the chocolate pudding in the basket: drift 0.00 cm; reach (cm) above chocolate pudding 2.4, basket 2.7; 50 resets for the start states
- put the book on the storage box and put the bbq sauce in the tray: drift 0.00 cm; reach (cm) above book 2.3, storage box 2.5, above bbq sauce 2.9, tray 2.4; 50 resets for the start states

Rejected layouts (last few): #14 lift the milk: {'drift': 1.0, 'low': -0.4382472434445275, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #15 put the chocolate pudding in the basket: {'drift': 1.3877787807814457e-14, 'low': -0.009352354195647394, 'start_success': False, 'forced': False, 'view': True, 'reach': {'above chocolate pudding': 0.063, 'basket': 0.028}}; #16 lift the milk: {'drift': 1.0, 'low': -0.43593038784347704, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #17 put the chocolate pudding in the basket: {'drift': 6.938893903907228e-15, 'low': -0.009352354195647394, 'start_success': False, 'forced': False, 'view': True, 'reach': {'above chocolate pudding': 0.06, 'basket': 0.025}}; #18 lift the milk: {'drift': 6.668925906876251e-11, 'low': -0.43651730399148997, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}

## S16 (coffee table): built after 1 layout(s), 144 s

Classes: basket = basket, cookies = cookies, milk = milk, orange juice = orange_juice, shelf = wooden_two_layer_shelf, storage box = white_storage_box.

- put the milk on the storage box: drift 0.00 cm; reach (cm) above milk 2.8, storage box 2.8; 50 resets for the start states
- put the cookies in the basket and put the orange juice on the shelf: drift 0.00 cm; reach (cm) above cookies 2.7, basket 2.7, above orange juice 2.7, shelf 2.6; 50 resets for the start states

## S17 (coffee table): built after 1 layout(s), 177 s

Classes: basket = basket, bookcase = wooden_shelf, butter = butter, macaroni = macaroni_and_cheese, plate = plate, ramekin = glazed_rim_porcelain_ramekin, storage box = white_storage_box, tomato sauce = tomato_sauce.

- put the butter in the bookcase and put the ramekin on the plate: drift 0.00 cm; reach (cm) above butter 2.2, bookcase opening 2.6, above ramekin 2.5, plate 2.5; 50 resets for the start states
- put the tomato sauce on the storage box and put the macaroni in the basket: drift 0.00 cm; reach (cm) above tomato sauce 2.7, storage box 2.8, above macaroni 2.7, basket 3.0; 50 resets for the start states

## S18 (tabletop): built after 1 layout(s), 264 s

Classes: bookcase = wooden_shelf, bowl = akita_black_bowl, cookies = cookies, macaroni = macaroni_and_cheese, orange juice = orange_juice, placemat = dining_set_group, ramekin = glazed_rim_porcelain_ramekin, storage box = white_storage_box.

- lift the bowl: drift 0.00 cm; reach (cm) above bowl 2.5; 50 resets for the start states
- put the orange juice on the storage box: drift 0.00 cm; reach (cm) above orange juice 2.7, storage box 2.6; 50 resets for the start states
- put the ramekin on the placemat: drift 0.00 cm; reach (cm) above ramekin 2.9, placemat 2.4; 50 resets for the start states
- put the cookies in the bowl and put the macaroni in the bookcase: drift 0.00 cm; reach (cm) above cookies 2.6, bowl 2.7, above macaroni 2.5, bookcase opening 3.0; 50 resets for the start states

## S19 (tabletop): built after 1 layout(s), 187 s

Classes: alphabet soup = alphabet_soup, basket = basket, bbq sauce = bbq_sauce, book = yellow_book, bookcase = wooden_shelf, shelf = wooden_two_layer_shelf, wine bottle = wine_bottle.

- put the book in the bookcase: drift 0.00 cm; reach (cm) above book 2.9, bookcase opening 3.1; 50 resets for the start states
- put the wine bottle in front of the basket: drift 0.00 cm; reach (cm) above wine bottle 2.7, beside 2.4; 50 resets for the start states
- put the bbq sauce in the basket and put the alphabet soup on the shelf: drift 0.00 cm; reach (cm) above bbq sauce 2.1, basket 2.2, above alphabet soup 2.8, shelf 2.3; 50 resets for the start states

## S20 (tabletop): built after 2 layout(s), 63 s

Classes: placemat = dining_set_group, plate = plate, shelf = wooden_two_layer_shelf, wine bottle = wine_bottle.

- put the plate on the placemat and put the wine bottle on the shelf: drift 0.00 cm; reach (cm) above plate 2.4, placemat 2.5, above wine bottle 2.2, shelf 2.4; 50 resets for the start states

Rejected layouts (last few): #0 put the plate on the placemat and put the wine bottle on the shelf: {'drift': 3.122502256758253e-17, 'low': -0.0014568679706069565, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S21 (kitchen): built after 4 layout(s), 155 s

Classes: alphabet soup = alphabet_soup, bowl = white_bowl, butter = butter, cabinet = white_cabinet, microwave = microwave.

- put the bowl in the microwave and close it: drift 0.00 cm; reach (cm) above bowl 2.7, microwave opening 2.5; 50 resets for the start states
- put the butter in the bowl and put the alphabet soup on the cabinet: drift 0.00 cm; reach (cm) above butter 2.8, bowl 2.7, above alphabet soup 3.0, cabinet top 2.9; 50 resets for the start states

Rejected layouts (last few): #0 put the bowl in the microwave and close it: {'drift': 1.0, 'low': -0.014097219264647465, 'start_success': False, 'forced': False, 'view': True, 'reach': {}}; #1 put the butter in the bowl and put the alphabet soup on the cabinet: {'drift': 0.0, 'low': -0.004237494667114539, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}; #2 put the butter in the bowl and put the alphabet soup on the cabinet: {'drift': 0.0, 'low': -0.004237494667114539, 'start_success': False, 'forced': False, 'view': False, 'reach': {}}

## S22 (tabletop): built after 1 layout(s), 117 s

Classes: alphabet soup = alphabet_soup, cabinet = wooden_cabinet, moka pot = moka_pot, shelf = wooden_two_layer_shelf, tomato sauce = tomato_sauce.

- open the middle drawer and put the alphabet soup in it: drift 0.00 cm; reach (cm) above alphabet soup 2.5, drawer front 2.4; 50 resets for the start states
- put the tomato sauce on the shelf and put the moka pot on the cabinet: drift 0.00 cm; reach (cm) above tomato sauce 2.7, shelf 2.8, above moka pot 2.6, cabinet top 2.4; 50 resets for the start states

## S23 (living room): built after 15 layout(s), 118 s

Classes: basket = basket, bookcase = wooden_shelf, frying pan = chefmate_8_frypan, mug = porcelain_mug, popcorn = popcorn.

- put the popcorn in the basket: drift 0.00 cm; reach (cm) above popcorn 2.8, basket 2.4; 51 resets for the start states
- put the mug in the bookcase and put the frying pan to the right of the basket: drift 0.00 cm; reach (cm) above mug 2.8, bookcase opening 2.9, above frying pan 2.7, beside 2.9; 50 resets for the start states

Rejected layouts (last few): #9 layout: no non-overlapping layout; #10 layout: no non-overlapping layout; #11 layout: no non-overlapping layout; #12 layout: no non-overlapping layout; #13 layout: no non-overlapping layout

## S24 (floor): built after 1 layout(s), 128 s

Classes: bowl = akita_black_bowl, cream cheese = cream_cheese, ketchup = ketchup, macaroni = macaroni_and_cheese, plate = plate, popcorn = popcorn, tray = wooden_tray.

- put the cream cheese on the plate and put the macaroni in the tray: drift 0.00 cm; reach (cm) above cream cheese 2.4, plate 2.8, above macaroni 2.4, tray 2.4; 50 resets for the start states
- put the popcorn in the bowl and put the ketchup behind the plate: drift 0.00 cm; reach (cm) above popcorn 2.3, bowl 2.4, above ketchup 2.2, beside 2.4; 50 resets for the start states

## S25 (kitchen): built after 1 layout(s), 90 s

Classes: butter = butter, frying pan = chefmate_8_frypan, microwave = microwave, ramekin = glazed_rim_porcelain_ramekin, stove = flat_stove.

- put the ramekin in the microwave: drift 0.00 cm; reach (cm) above ramekin 2.9, microwave opening 2.9; 50 resets for the start states
- turn on the stove and put the butter in the pan: drift 0.00 cm; reach (cm) above butter 2.7, pan 2.8; 50 resets for the start states
