import React, { useState } from "react"
import {Container, Col, Row, Modal, ModalBody, Button, Tooltip, OverlayTrigger, Navbar, Nav} from 'react-bootstrap';
import Chessground from "react-chessground"

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faRedoAlt, faChessBoard } from '@fortawesome/free-solid-svg-icons'

import { Scrollbars } from 'rc-scrollbars';


import * as ChessJS from "chess.js";
const Chess = typeof ChessJS === "function" ? ChessJS : ChessJS.Chess;
const axios = require('axios');


export default () => {

    const [chess, setChess] = useState(new Chess())
    const [pendingMove, setPendingMove] = useState()
    const [selectVisible, setSelectVisible] = useState(false)
    const [gameOver, setGameOver] = useState(false)
    const [fen, setFen] = useState("")
    const [lastMove, setLastMove] = useState()
    const [boardOrientation, setBoardOrientation] = useState("white")
    const [enterPosition, setEnterPosition] = useState(false)
    const [startColorVisible, setStartColorVisible] = useState(true)
    const [startColor, setStartColor] = useState("")
    const [isViewOnly, setIsViewOnly] = useState(false)

    const onMove = (from, to) => {
        const moves = chess.moves({ verbose: true })
        for (let i = 0, len = moves.length; i < len; i++) { /* eslint-disable-line */
            if (moves[i].flags.indexOf("p") !== -1 && moves[i].from === from) {
                setPendingMove([from, to])
                setSelectVisible(true)
                return
            }
        }
        if (chess.move({ from, to, promotion: "x" })) {
            setFen(chess.fen())
            setLastMove([from, to])
            if (chess.game_over()) setGameOver(true)
            setTimeout(randomMove, 500)
        }
    }

    const randomMove = () => {
        const fen = chess.fen()
        axios.post('/api/move', {"fen": fen})
        .then(response => {
            chess.move(response.data.move, { sloppy:true })
            setFen(chess.fen())
            if (chess.game_over()) setGameOver(true)
            setLastMove([response.data.move_from, response.data.move_to])
        });
    }

    const promotion = e => {
        const from = pendingMove[0]
        const to = pendingMove[1]
        chess.move({ from, to, promotion: e })
        setFen(chess.fen())
        setLastMove([from, to])
        if (chess.game_over()) setGameOver(true)
        setSelectVisible(false)
        setTimeout(randomMove, 500)
    }

    const turnColor = () => {
        return chess.turn() === "w" ? "white" : "black"
    }

    const calcMovable = () => {
        const dests = new Map()
        chess.SQUARES.forEach(s => {
            const ms = chess.moves({ square: s, verbose: true })
            if (ms.length) dests.set(s, ms.map(m => m.to))
        })
        return {
            free: false,
            dests,
            color: chess.turn() === "w" ? "white" : "black"
        }
    }

    const resignGame = () => {
        const colr = startColor === "w" ? "white" : "black"
        chess.set_comment(`${colr} resigns`)
        setIsViewOnly(true)
    }

    const newGame = () => {
        chess.reset()
        setLastMove(null)
        setFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        setIsViewOnly(false)
        setStartColorVisible(true)
    }

    const injectGA = () => {
        if (typeof window == 'undefined') {
            return;
        }
        window.dataLayer = window.dataLayer || [];
        function gtag() {
            window.dataLayer.push(arguments);
        }
        gtag('js', new Date());

        gtag('config', 'G-LMH3PFTZZP');
    };

    const pgnMoveHistory = chess.pgn({max_width: 5, newline_char: '<br />'})

    const tabulatedpgnMoveHistory = (pgnMoveHistoryString) =>{

        var pgnMoveHistoryArray= pgnMoveHistoryString.split('<br />');
        var htmlString=""
        if(pgnMoveHistoryArray.length>=2)
        {
            htmlString=htmlString+pgnMoveHistoryArray[0]+'<br />'+pgnMoveHistoryArray[1]+'<br />'+'<br />';
            if(pgnMoveHistoryArray.length>3){
                if (pgnMoveHistoryArray[2].startsWith('[') )
                    htmlString=htmlString+pgnMoveHistoryArray[2]+'<br />';
                if (pgnMoveHistoryArray[3].startsWith('[') )
                    htmlString=htmlString+pgnMoveHistoryArray[3]+'<br />';
                htmlString=htmlString+'<table style="box-shadow: 0 2px 2px 0 rgb(0 0 0 / 14%), 0 3px 1px -2px rgb(0 0 0 / 20%), 0 1px 5px 0 rgb(0 0 0 / 12%); border-collapse: collapse; font-family: sans-serif;table-layout: fixed;width: 100%;">';
            }

            for (var i = 3; i < pgnMoveHistoryArray.length; i++) { 
                htmlString=htmlString+'<tr>';
                var temp=pgnMoveHistoryArray[i].split(" ");
                console.log(temp.length);
                console.log(temp[0]);
                console.log(temp[1]);
                htmlString=htmlString+'<td style="width: 30px;text-align:center; background: #dcdcdc">'+temp[0].slice(0, -1)+'</td>';
                htmlString=htmlString+'<td style="padding-left: 0.7em; background: #f7f6f5; cursor: pointer">'+temp[1]+'</td>';
                if(temp.length==3)
                {
                    console.log(temp[2]);
                    htmlString=htmlString+'<td style="padding-left: 0.7em; background: #f7f6f5; cursor: pointer">'+temp[2]+'</td>';
                }
                else{
                    htmlString=htmlString+'<td style="padding-left: 0.7em; background: #f7f6f5; cursor: pointer">'+""+'</td>';
                }
                
                htmlString=htmlString+'</tr>';
            }

            if(pgnMoveHistoryArray.length>3){
                htmlString=htmlString+'</table>';
            }
        }

        return htmlString;
    }

    const downloadPGNFile = () => {
        if (process.browser) {
            const element = document.createElement("a");
            const file = new Blob([chess.pgn()], {type: 'text/plain'});
            element.href = URL.createObjectURL(file);
            element.download = "myGame.pgn";
            document.body.appendChild(element); // Required for this to work in FireFox
            element.click();
        }
    }

    const renderFlipTooltip = (props) => (
        <Tooltip id="button-tooltip" {...props}>
            Flip Board
        </Tooltip>
    );

    const renderFENTooltip = (props) => (
        <Tooltip id="button-tooltip" {...props}>
            Enter FEN
        </Tooltip>
    );

    const onPositionSubmit = () => {
        chess.load(fen)
        setEnterPosition(false);
    }

    return (

        <Container fluid style={{ background: "#e2dfdb", height: "100vh" }}>
            <Navbar>
                <Navbar.Brand href="#home">TrojanKnights</Navbar.Brand>
                <Nav className="mr-auto">
                    <Nav.Link href="#newgame" onClick={() => newGame()}>New Game</Nav.Link>
                    {/* <Nav.Link href="#features">Features</Nav.Link>
                <Nav.Link href="#pricing">Pricing</Nav.Link> */}
                </Nav>
            </Navbar>
            <Modal show={enterPosition}>
                <ModalBody>
                    <form >
                        <label>
                            Position as FEN string
                            <input
                                type="text"
                                value={fen}
                                onChange={e => setFen(e.target.value)}
                            />
                        </label>
                        <input onClick={() => onPositionSubmit()} type="button" value="Submit" />
                    </form>

                </ModalBody>
            </Modal>
            <Modal show={startColorVisible} backdrop="static">
                <Modal.Header>
                    <Modal.Title>Play As</Modal.Title>
                </Modal.Header>
                <ModalBody>
                    <input className={"ml-2"} onClick={() => {setStartColor("w"); setStartColorVisible(false); setBoardOrientation("white"); chess.header('White','Human'); chess.header('Black','AI')}} type="button" value="White" />
                    <input className={"ml-3"} onClick={() => {setStartColor("b"); setStartColorVisible(false); setBoardOrientation("black"); chess.header('White','AI'); chess.header('Black','Human'); setTimeout(randomMove, 500); }} type="button" value="Black" />
                </ModalBody>
            </Modal>
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-LMH3PFTZZP"></script>
            <script>{injectGA()}</script>
            <Row>
                <Col md={2}/>
                <Col md={6}>
                    <Modal
                        show={selectVisible}
                    >
                        <ModalBody>
                            <div style={{ textAlign: "center", cursor: "pointer" }}>
          <span role="presentation" onClick={() => promotion("q")}>
            <img src="images/wQ.svg" alt="" style={{ width: 100 }} />
          </span>
                                <span role="presentation" onClick={() => promotion("r")}>
            <img src="images/wR.svg" alt="" style={{ width: 100 }} />
          </span>
                                <span role="presentation" onClick={() => promotion("b")}>
            <img src="images/wB.svg" alt="" style={{ width: 100 }} />
          </span>
                                <span role="presentation" onClick={() => promotion("n")}>
            <img src="images/wN.svg" alt="" style={{ width: 100 }} />
          </span>
                            </div>
                        </ModalBody>
                    </Modal>
                    <div className={"mt-3"}>
                        <Chessground
                            width="40vw"
                            height="40vw"
                            turnColor={turnColor()}
                            movable={calcMovable()}
                            lastMove={lastMove}
                            fen={fen}
                            onMove={onMove}
                            orientation={boardOrientation}
                            viewOnly={isViewOnly}
                        />
                        <OverlayTrigger
                            placement="bottom"
                            delay={{ show: 250, hide: 400 }}
                            overlay={renderFlipTooltip}
                        >
                        <FontAwesomeIcon className={"mt-3"} icon={faRedoAlt} size="2x" onClick={() => boardOrientation === "white" ? setBoardOrientation("black") : setBoardOrientation("white")} />
                        </OverlayTrigger>
                        <OverlayTrigger
                            placement="bottom"
                            delay={{ show: 250, hide: 400 }}
                            overlay={renderFENTooltip}
                        >
                        <FontAwesomeIcon className={"mt-3 ml-4"} icon={faChessBoard} size="2x" onClick={() => setEnterPosition(true)} />
                        </OverlayTrigger>
                    </div>
                </Col>
                <Col md={4} className={"mt-5"}>
                    
                    <Scrollbars>
                        <div style={{height: "35vw"}} dangerouslySetInnerHTML={{__html: tabulatedpgnMoveHistory(pgnMoveHistory)}}></div>
                    </Scrollbars> 

                    <Button onClick={() => downloadPGNFile()}>Download PGN File</Button>
                    <Button className={"ml-1"} onClick={() => resignGame()}>Resign</Button>
                    <Button className={"ml-1"} onClick={() => newGame()}>New Game</Button>
                </Col>
            </Row>
        </Container>
    )
}
